import logging
from enum import Enum
from typing import Dict, Optional, Tuple

import accelerate
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm

from models.bet.gpt import GPT
from models.bet.utils import MLP


GENERATOR_SEED_FIXED = 123456789


class KMeansDiscretizer:
    """
    Simplified and modified version of KMeans algorithm from sklearn.
    We initialize this with a fixed seed to ensure that on each GPU we come up with the same
    clusters.
    """

    def __init__(
        self,
        num_bins: int = 100,
        kmeans_iters: int = 50,
    ):
        super().__init__()
        self.n_bins = num_bins
        self.kmeans_iters = kmeans_iters

    def fit(self, input_actions: torch.Tensor) -> None:
        self.bin_centers = KMeansDiscretizer._kmeans(
            input_actions, ncluster=self.n_bins, niter=self.kmeans_iters
        )

    @classmethod
    def _kmeans(cls, x: torch.Tensor, ncluster: int = 512, niter: int = 50):
        """
        Simple k-means clustering algorithm adapted from Karpathy's minGPT libary
        https://github.com/karpathy/minGPT/blob/master/play_image.ipynb
        """
        N, D = x.size()
        generator = torch.Generator()
        generator.manual_seed(GENERATOR_SEED_FIXED)

        c = x[
            torch.randperm(N, generator=generator)[:ncluster]
        ]  # init clusters at random, with a fixed seed

        pbar = tqdm.trange(niter)
        pbar.set_description("K-means clustering")
        for i in pbar:
            # assign all pixels to the closest codebook element
            a = ((x[:, None, :] - c[None, :, :]) ** 2).sum(-1).argmin(1)
            # move each codebook element to be the mean of the pixels that assigned to it
            c = torch.stack([x[a == k].mean(0) for k in range(ncluster)])
            # re-assign any poorly positioned codebook elements
            nanix = torch.any(torch.isnan(c), dim=1)
            ndead = nanix.sum().item()
            if ndead:
                tqdm.tqdm.write(
                    "done step %d/%d, re-initialized %d dead clusters"
                    % (i + 1, niter, ndead)
                )
            c[nanix] = x[
                torch.randperm(N, generator=generator)[:ndead]
            ]  # re-init dead clusters
        return c


class BehaviorTransformer(nn.Module):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        gpt_model: GPT,
        num_extra_predicted_actions: Optional[int] = None,
        trainable_obs_padding: bool = True,
        n_clusters: int = 32,
        kmeans_fit_steps: int = 500,
        kmeans_iters: int = 50,
        offset_loss_multiplier: float = 1.0e3,
        offset_distance_metric: str = "L2",
        gamma: float = 2.0,
        **kwargs,
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self._goal_dim = goal_dim
        self._num_extra_predicted_actions = num_extra_predicted_actions
        # Gradient-free, all zeros if we don't want to train this.
        self._obs_padding = nn.Parameter(
            trainable_obs_padding * torch.randn(obs_dim),
            requires_grad=trainable_obs_padding,
        )

        if goal_dim <= 0:
            self._cbet_method = self.GOAL_SPEC.unconditional
        elif obs_dim == goal_dim:
            self._cbet_method = self.GOAL_SPEC.concat
        else:
            self._cbet_method = self.GOAL_SPEC.stack

        self._gpt_model = gpt_model
        # For now, we assume the number of clusters is given.
        assert n_clusters > 0
        self._K = n_clusters
        self._kmeans_fit_steps = kmeans_fit_steps
        self._clustering_algo = KMeansDiscretizer(
            num_bins=n_clusters, kmeans_iters=kmeans_iters
        )
        self._current_steps = 0
        self._map_to_cbet_preds = MLP(
            in_channels=gpt_model.config.output_dim,
            hidden_channels=[(act_dim + 1) * n_clusters],
        )
        self._collected_actions = []
        self._have_fit_kmeans = False
        self._offset_loss_multiplier = offset_loss_multiplier
        # Placeholder for the cluster centers.
        generator = torch.Generator()
        generator.manual_seed(GENERATOR_SEED_FIXED)
        self.register_buffer(
            "_cluster_centers",
            torch.randn(
                (n_clusters, act_dim), generator=generator, dtype=torch.float32
            ),
        )
        self._criterion = FocalLoss(gamma=gamma, reduction="none")
        self._offset_criterion = (
            nn.MSELoss(reduction="none")
            if offset_distance_metric == "L2"
            else nn.L1Loss(reduction="none")
        )
        self._accelerator = accelerate.Accelerator()

    def _load_from_state_dict(self, *args, **kwargs):
        # Don't fit kmeans if we are loading from a state dict.
        self._current_steps = self._kmeans_fit_steps
        self._have_fit_kmeans = True
        return super()._load_from_state_dict(*args, **kwargs)

    def forward(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        padding_seq: Optional[torch.Tensor],
        predict_with_offset: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        if self._current_steps == 0:
            self._cluster_centers = self._cluster_centers.to(obs_seq.device)
        if self._current_steps < self._kmeans_fit_steps and action_seq is not None:
            self._current_steps += 1
            self._fit_kmeans(obs_seq, goal_seq, action_seq, padding_seq)
        return self._predict(
            obs_seq,
            goal_seq,
            action_seq,
            padding_seq,
            predict_with_offset=predict_with_offset,
        )

    def _fit_kmeans(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        padding_seq: Optional[torch.Tensor],
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        assert self._current_steps <= self._kmeans_fit_steps
        if self._current_steps == 1:
            self._cluster_centers = self._cluster_centers.to(action_seq.device)

        all_action_seq = self._accelerator.gather(action_seq)
        all_padding_seq = self._accelerator.gather(padding_seq)
        self._collected_actions.append(
            all_action_seq[torch.logical_not(all_padding_seq)]
        )
        if self._current_steps == self._kmeans_fit_steps:
            logging.info("Fitting KMeans")
            self._clustering_algo.fit(
                torch.cat(self._collected_actions, dim=0).view(-1, self._act_dim)
            )
            self._have_fit_kmeans = True
            self._cluster_centers = self._clustering_algo.bin_centers.float().to(
                action_seq.device
            )

    def _predict(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: Optional[torch.Tensor],
        is_padded_action_seq: Optional[torch.Tensor],
        predict_with_offset: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        batch_size, obs_T, _ = obs_seq.shape
        _, action_T, _ = (
            action_seq.shape if action_seq is not None else (None, None, None)
        )
        # Take the one that is not None.
        actions_to_predict = action_T or obs_T
        if self._num_extra_predicted_actions:
            actions_to_predict += self._num_extra_predicted_actions
        # Now, figure out if we should pad the obs seq.
        if obs_T < actions_to_predict:
            # We need to pad the obs seq.
            pad_size = actions_to_predict - obs_T
            padded_obs_seq = torch.cat(
                [
                    obs_seq,
                    einops.repeat(
                        self._obs_padding, "D -> N T D", N=batch_size, T=pad_size
                    ),
                ],
                dim=1,
            )
        else:
            padded_obs_seq = obs_seq
        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        if self._cbet_method == self.GOAL_SPEC.unconditional:
            gpt_input = padded_obs_seq
        elif self._cbet_method == self.GOAL_SPEC.concat:
            gpt_input = torch.cat([goal_seq, padded_obs_seq], dim=1)
        elif self._cbet_method == self.GOAL_SPEC.stack:
            gpt_input = torch.cat([goal_seq, padded_obs_seq], dim=-1)
        else:
            raise NotImplementedError

        gpt_output = self._gpt_model(gpt_input)
        if self._cbet_method == self.GOAL_SPEC.concat:
            # Chop off the goal encodings.
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        cbet_preds = self._map_to_cbet_preds(gpt_output)
        cbet_logits, cbet_offsets = torch.split(
            cbet_preds, [self._K, self._K * self._act_dim], dim=-1
        )
        cbet_offsets = einops.rearrange(cbet_offsets, "N T (K A) -> N T K A", K=self._K)

        cbet_probs = torch.softmax(cbet_logits, dim=-1)
        N, T, choices = cbet_probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_centers = einops.rearrange(
            torch.multinomial(cbet_probs.view(-1, choices), num_samples=1),
            "(N T) 1 -> N T 1",
            N=N,
        )
        flattened_cbet_offsets = einops.rearrange(cbet_offsets, "N T K A -> (N T) K A")
        sampled_offsets = flattened_cbet_offsets[
            torch.arange(flattened_cbet_offsets.shape[0]), sampled_centers.flatten()
        ].view(N, T, self._act_dim)
        centers = self._cluster_centers[sampled_centers.flatten()].view(
            N, T, self._act_dim
        )
        a_hat = centers
        if predict_with_offset:
            a_hat += sampled_offsets
        if action_seq is None:
            return a_hat, None, {}
        # We are in training, so figure out the loss for the actions.
        # First, we need to find the closest cluster center for each action.
        action_bins = self._find_closest_cluster(action_seq)
        true_offsets = action_seq - self._cluster_centers[action_bins]
        predicted_offsets = flattened_cbet_offsets[
            torch.arange(flattened_cbet_offsets.shape[0]), action_bins.flatten()
        ].view(N, T, self._act_dim)
        # Now we can compute the loss.
        offset_loss = self._offset_criterion(predicted_offsets, true_offsets)
        cbet_loss = self._criterion(
            einops.rearrange(cbet_logits, "N T D -> (N T) D"),
            einops.rearrange(action_bins, "N T -> (N T)"),
        )
        # Now, use the padding mask to mask out the loss.
        if is_padded_action_seq is not None:
            cbet_loss *= ~is_padded_action_seq.view(-1)
            offset_loss *= ~is_padded_action_seq.unsqueeze(-1)
        cbet_loss, offset_loss = cbet_loss.mean(), offset_loss.mean()
        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
        action_mse = F.mse_loss(a_hat, action_seq, reduction="none")
        action_l1 = F.l1_loss(a_hat, action_seq, reduction="none")
        norm = torch.norm(action_seq, p=2, dim=-1, keepdim=True) + 1e-9
        normalized_mse = (action_mse / norm).mean()
        if self._current_steps < self._kmeans_fit_steps:
            loss = loss.detach() + (loss * 0.0)

        loss_dict = {
            "classification_loss": cbet_loss.detach().cpu().item(),
            "offset_loss": offset_loss.detach().cpu().item(),
            "loss": loss.detach().cpu().item(),
            "L2_loss": action_mse.mean().detach().cpu().item(),
            "L2_loss_normalized": normalized_mse.mean().detach().cpu().item(),
            "L1_loss": action_l1.mean().detach().cpu().item(),
        }
        return a_hat, loss, loss_dict

    def _find_closest_cluster(self, action_seq: torch.Tensor) -> torch.Tensor:
        N, T, _ = action_seq.shape
        flattened_actions = einops.rearrange(action_seq, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (flattened_actions[:, None, :] - self._cluster_centers[None, :, :]) ** 2,
            dim=2,
        )  # (N T) K A -> (N T) K
        closest_cluster_center = torch.argmin(cluster_center_distance, dim=1)  # (N T)
        discretized_action = einops.rearrange(
            closest_cluster_center, "(N T) -> N T", N=N, T=T
        )
        return discretized_action

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        optimizer = self._gpt_model.configure_optimizers(
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
        )
        optimizer.add_param_group({"params": self._map_to_cbet_preds.parameters()})
        return optimizer


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
