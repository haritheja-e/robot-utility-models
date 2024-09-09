import logging
from enum import Enum
from typing import Dict, Optional, Tuple

import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from models.bet.gpt import GPT
from models.bet.utils import MLP
from models.bet.vqvae.vqvae import VqVae

GENERATOR_SEED_FIXED = 123456789


class VQBehaviorTransformer(nn.Module):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        gpt_model: GPT,
        vqvae_model: VqVae,
        offset_loss_multiplier: float = 1.0e2,
        secondary_code_multiplier: float = 0.5,
        gamma: float = 2.0,
        obs_window_size=10,
        act_window_size=10,
        sequentially_select=False,
        use_og_bet_loss=False,
        use_half_and_half_loss=False,
        temperature=1.0,
        device="cuda",
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._goal_dim = goal_dim
        self.obs_window_size = obs_window_size
        self.act_window_size = act_window_size
        self.sequentially_select = sequentially_select
        self._use_og_bet_loss = use_og_bet_loss
        self._use_half_and_half_loss = use_half_and_half_loss
        self.temperature = temperature
        self.device = device

        if goal_dim <= 0:
            self._cbet_method = self.GOAL_SPEC.unconditional
        elif obs_dim == goal_dim:
            self._cbet_method = self.GOAL_SPEC.concat
        else:
            self._cbet_method = self.GOAL_SPEC.stack

        self._gpt_model = gpt_model
        self._vqvae_model = vqvae_model
        self._G = self._vqvae_model.vqvae_groups  # G(number of groups)
        self._C = self._vqvae_model.vqvae_n_embed  # C(number of code integers)
        self._D = self._vqvae_model.embedding_dim  # D(embedding dims)
        self._current_steps = 0
        if self.sequentially_select:
            print("use sequantial prediction for vq dictionary!")
            self._map_to_cbet_preds_bin1 = MLP(
                in_channels=gpt_model.config.output_dim,
                hidden_channels=[512, 512, self._C],
            )
            self._map_to_cbet_preds_bin2 = MLP(
                in_channels=gpt_model.config.output_dim + self._C,
                hidden_channels=[512, self._C],
            )
        else:
            self._map_to_cbet_preds_bin = MLP(
                in_channels=gpt_model.config.output_dim,
                hidden_channels=[1024, 1024, self._G * self._C],
            )
        self._map_to_cbet_preds_offset = MLP(
            in_channels=gpt_model.config.output_dim,
            hidden_channels=[
                1024,
                1024,
                self._G * self._C * (act_dim * self.act_window_size),
            ],
        )
        self._offset_loss_multiplier = offset_loss_multiplier
        self._secondary_code_multiplier = secondary_code_multiplier
        self._criterion = FocalLoss(gamma=gamma)

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        second_half: bool = False,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        # VQ-BeT doesn't use "padding_seq" and "predict_with_offset" input
        return self._predict(obs_seq, goal_seq, action_seq, second_half)

    def _predict(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        second_half: bool,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        """
        Assume dimensions are N T D for N sequences of T timesteps with dimension D.

        obs_seq : (Batch size) X (obs window) X (obs dim)
        goal_seq : (Batch size) X (goal window) X (obs dim)
        action_seq : (Batch size) X (total window) X (action dim)

        assume that
        total_w (total window) of action sequence is
        total_w = obs_w + act_w - 1
        e.g. is observation window is 5, and antion pred window is 3,
        the obs seq (o_{t-4}, o_{t-3}, o_{t-2}, o_{t-1}, o_{t}) will predict (a_{t}, a_{t+1}, a_{t+2})

        However, we not only predict (a_{t}, a_{t+1}, a_{t+2}), but also all the action seq for all the actions in the sequence

        o_{t-4} -> |   | -> (a_{t-4}, a_{t-3}, a_{t-2})
        o_{t-3} -> | G | -> (a_{t-3}, a_{t-2}, a_{t-1})
        o_{t-2} -> | P | -> (a_{t-2}, a_{t-1}, a_{t})
        o_{t-1} -> | T | -> (a_{t-1}, a_{t}, a_{t+1})
        o_{t}   -> |   | -> (a_{t}, a_{t+1}, a_{t+2})
        """
        if obs_seq.shape[1] < self.obs_window_size:
            # if input size is smaller than obs_window size (e.g. the initial steps of env eval episodes,
            # VQ-BeT copy the obs and tile it to match obs_window_size
            obs_seq = torch.cat(
                (
                    torch.tile(
                        obs_seq[:, 0, :],
                        (1, self.obs_window_size - obs_seq.shape[1], 1),
                    ),
                    obs_seq,
                ),
                dim=-2,
            )
        if self._cbet_method == self.GOAL_SPEC.unconditional:
            gpt_input = obs_seq
        elif self._cbet_method == self.GOAL_SPEC.concat:
            gpt_input = torch.cat([goal_seq, obs_seq], dim=1)
        elif self._cbet_method == self.GOAL_SPEC.stack:
            gpt_input = torch.cat([goal_seq, obs_seq], dim=-1)
        else:
            raise NotImplementedError

        gpt_output = self._gpt_model(gpt_input)
        if self._cbet_method == self.GOAL_SPEC.concat:
            # Chop off the goal encodings.
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        gpt_output = einops.rearrange(gpt_output, "N T (G C) -> (N T) (G C)", G=self._G)
        obs = einops.rearrange(obs_seq, "N T O -> (N T) O")
        obs = obs.unsqueeze(dim=1)
        # note that output of offset network is G C WA,
        # where G is number of 'layers' of Residual VQ-VAE
        # C is number of words in each layer's dictionary
        # and W, A is predicted action window, and predicted action dims
        if self.sequentially_select:
            cbet_logits1 = self._map_to_cbet_preds_bin1(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs1 = torch.softmax(cbet_logits1 / self.temperature, dim=-1)
            NT, choices = cbet_probs1.shape
            G = self._G
            sampled_centers1 = einops.rearrange(
                torch.multinomial(cbet_probs1.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(sampled_centers1, num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_probs2 = torch.softmax(cbet_logits2 / self.temperature, dim=-1)
            sampled_centers2 = einops.rearrange(
                torch.multinomial(cbet_probs2.view(-1, choices), num_samples=1),
                "(NT) 1 -> NT",
                NT=NT,
            )
            sampled_centers = torch.stack(
                (sampled_centers1, sampled_centers2), axis=1
            )  # NT, G
        else:
            cbet_logits = self._map_to_cbet_preds_bin(gpt_output)
            cbet_offsets = self._map_to_cbet_preds_offset(gpt_output)
            cbet_logits = einops.rearrange(
                cbet_logits, "(NT) (G C) -> (NT) G C", G=self._G
            )
            cbet_offsets = einops.rearrange(
                cbet_offsets, "(NT) (G C WA) -> (NT) G C WA", G=self._G, C=self._C
            )
            cbet_probs = torch.softmax(cbet_logits / self.temperature, dim=-1)
            NT, G, choices = cbet_probs.shape
            sampled_centers = einops.rearrange(
                torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
                "(NT G) 1 -> NT G",
                NT=NT,
            )
        if action_seq is not None:
            n, total_w, act_dim = action_seq.shape
            act_w = self._vqvae_model.input_dim_h
            obs_w = total_w + 1 - act_w
            output_shape = (n, obs_w, act_w, act_dim)
            output = torch.empty(output_shape).to(action_seq.device)
            for i in range(obs_w):
                output[:, i, :, :] = action_seq[:, i : i + act_w, :]
            action_seq = einops.rearrange(output, "N T W A -> (N T) W A")
            _, action_bins = self._vqvae_model.get_code(
                action_seq, obs
            )  # action_bins: NT, G

        with torch.no_grad():
            centers = self._vqvae_model.draw_code_forward(sampled_centers).view(
                NT, -1, self._D
            )
            return_decoder_input = einops.rearrange(
                centers.clone().detach(), "NT G D -> NT (G D)"
            )
            decoded_action = (
                self._vqvae_model.get_action_from_latent(return_decoder_input, obs)
                .clone()
                .detach()
            )  # NT, A

        def get_offset_from_centers(centers):
            indices = (
                torch.arange(NT, device=self.device).unsqueeze(1),
                torch.arange(self._G, device=self.device).unsqueeze(0),
                centers,
            )
            # Use advanced indexing to sample the values
            sampled_offsets = cbet_offsets[indices]  # NT, G, WA or NT, G, A
            sampled_offsets = sampled_offsets.sum(dim=1)
            sampled_offsets = einops.rearrange(
                sampled_offsets, "NT (W A) -> NT W A", W=self._vqvae_model.input_dim_h
            )
            return sampled_offsets

        if self._use_og_bet_loss and action_seq is not None:
            sampled_offsets = get_offset_from_centers(action_bins)
        else:
            sampled_offsets = get_offset_from_centers(sampled_centers)

        a_hat = decoded_action + sampled_offsets

        if action_seq is None:
            return a_hat, None, {}
        # Figure out the loss for the actions.
        # First, we need to find the GT VQ codes for each action.

        # Now we can compute the loss.
        if action_seq.ndim == 2:
            action_seq = action_seq.unsqueeze(0)

        offset_target = action_seq - decoded_action
        if self._use_half_and_half_loss and action_seq is not None:
            offset_gt = get_offset_from_centers(action_bins)
            offset_sampled = get_offset_from_centers(sampled_centers)
            offset_loss = 0.5 * torch.nn.L1Loss()(
                offset_target, offset_gt
            ) + 0.5 * torch.nn.L1Loss()(offset_target, offset_sampled)
        else:
            offset_loss = torch.nn.L1Loss()(offset_target, sampled_offsets)

        if self.sequentially_select:
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits1[:, :],
                action_bins[:, 0],
            )
            cbet_logits2 = self._map_to_cbet_preds_bin2(
                torch.cat(
                    (gpt_output, F.one_hot(action_bins[:, 0], num_classes=self._C)),
                    axis=1,
                )
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits2[:, :],
                action_bins[:, 1],
            )
        else:
            cbet_loss1 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 0, :],
                action_bins[:, 0],
            )
            cbet_loss2 = self._criterion(  # F.cross_entropy
                cbet_logits[:, 1, :],
                action_bins[:, 1],
            )
        cbet_loss = cbet_loss1 * 5 + cbet_loss2 * self._secondary_code_multiplier

        equal_total_code_rate = (
            torch.sum(
                (torch.sum((action_bins == sampled_centers).int(), axis=1) == G).int()
            )
            / NT
        )
        equal_primary_code_rate = torch.sum(
            (action_bins[:, 0] == sampled_centers[:, 0]).int()
        ) / (NT)
        equal_secondary_code_rate = torch.sum(
            (action_bins[:, 1] == sampled_centers[:, 1]).int()
        ) / (NT)
        # if second_half:
        #     cbet_loss = cbet_loss * 0
        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        action_mse = F.mse_loss(a_hat, action_seq, reduction="none")
        action_l1 = F.l1_loss(a_hat, action_seq, reduction="none")
        norm = torch.norm(action_seq, p=2, dim=-1, keepdim=True) + 1e-9
        normalized_mse = (action_mse / norm).mean()

        translation_loss = F.mse_loss(a_hat[:, :, :3], action_seq[:, :, :3]).detach()
        rotation_loss = F.mse_loss(a_hat[:, :, 3:6], action_seq[:, :, 3:6]).detach()
        gripper_loss = F.mse_loss(a_hat[:, :, 6:], action_seq[:, :, 6:]).detach()

        loss_dict = {
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "loss": loss.detach().cpu().item(),
            "equal_total_code_rate": equal_total_code_rate,
            "equal_primary_code_rate": equal_primary_code_rate,
            "equal_secondary_code_rate": equal_secondary_code_rate,
            "L2_loss": action_mse.mean().detach().cpu().item(),
            "L2_loss_normalized": normalized_mse.mean().detach().cpu().item(),
            "L1_loss": action_l1.mean().detach().cpu().item(),
            "translation_loss": translation_loss,
            "rotation_loss": rotation_loss,
            "gripper_loss": gripper_loss,
        }
        return a_hat, loss, loss_dict

    # def configure_optimizers(self, weight_decay, learning_rate, betas):

    #     optimizer1 = self._gpt_model.configure_optimizers(
    #         weight_decay=weight_decay,
    #         learning_rate=learning_rate,
    #         betas=betas,
    #     )
    #     if self.sequentially_select:
    #         optimizer1.add_param_group({"params": self._map_to_cbet_preds_bin1.parameters()})
    #         optimizer1.add_param_group({"params": self._map_to_cbet_preds_bin2.parameters()})
    #     else:
    #         optimizer1.add_param_group({"params": self._map_to_cbet_preds_bin.parameters()})
    #     optimizer2 = torch.optim.AdamW(self._map_to_cbet_preds_offset.parameters(), lr=learning_rate, weight_decay=weight_decay, betas=betas)
    #     return {"optimizer1": optimizer1, "optimizer2": optimizer2}

    def _begin_epoch(self, optimizer, **kwargs):
        # log codebook usage rate for debugging
        # lr_0 = optimizer.param_groups[0]["lr"]
        # lr_neg1 = optimizer.param_groups[-1]["lr"]
        # return {"lr_0": lr_0, "lr_neg1": lr_neg1}
        return None

    def _load_from_state_dict(self, *args, **kwargs):
        # Don't fit kmeans if we are loading from a state dict.
        # if (path / "cbet_model.pt").exists():
        #     self.load_state_dict(torch.load(path / "cbet_model.pt"))
        # elif (path / "gpt_model.pt").exists():
        #     self._gpt_model.load_state_dict(torch.load(path / "gpt_model.pt"))
        # else:
        #     logging.warning("No model found at %s", path)
        return super()._load_from_state_dict(*args, **kwargs)

    # def load_model(self, path):
    #     if (path / "cbet_model.pt").exists():
    #         self.load_state_dict(torch.load(path / "cbet_model.pt"))
    #     elif (path / "gpt_model.pt").exists():
    #         self._gpt_model.load_state_dict(torch.load(path / "gpt_model.pt"))
    #     else:
    #         logging.warning("No model found at %s", path)


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 0, reduction: str = "mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        if reduction not in ("mean", "sum", "none"):
            raise NotImplementedError
        self.reduction = reduction

    def forward(self, input, target):
        logpt = F.log_softmax(input, dim=-1)
        logpt = logpt.gather(1, target.view(-1, 1)).view(-1)
        pt = logpt.exp()

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss
