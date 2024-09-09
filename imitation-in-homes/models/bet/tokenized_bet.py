import logging
from enum import Enum
from itertools import chain
from typing import Dict, Optional, Sequence, Tuple, Union

import accelerate
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.bet.bet import FocalLoss, KMeansDiscretizer
from models.bet.gpt import GPT
from models.bet.utils import MLP

GENERATOR_SEED_FIXED = 123456789


class TokenizedBehaviorTransformer(nn.Module):
    GOAL_SPEC = Enum("GOAL_SPEC", "concat stack unconditional")

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        goal_dim: int,
        gpt_model: GPT,
        action_spec: Optional[Sequence[int]] = None,
        start_and_ends: Optional[Sequence[Tuple[int, int]]] = None,
        num_extra_predicted_actions: Optional[int] = None,
        n_clusters: Union[int, Sequence[int]] = 32,
        kmeans_fit_steps: int = 500,
        kmeans_iters: int = 50,
        offset_loss_multiplier: float = 1.0e3,
        offset_distance_metric: str = "L2",
        representation_height: int = 7,
        representation_width: int = 7,
        gamma: float = 2.0,
        sampling_temperature: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        self._obs_dim = obs_dim
        self._act_dim = act_dim
        self.sampling_temperature = sampling_temperature
        # First ensure either action spec is given or start and ends are given.
        assert (action_spec is not None) != (
            start_and_ends is not None
        ), "EITHER action_spec OR start_and_ends must be given."
        if action_spec is not None:
            self._action_spec = action_spec
            assert act_dim == sum(action_spec)
            cumsum = [sum(action_spec[:i]) for i in range(len(action_spec) + 1)]
            self._starts = cumsum[:-1]
            self._ends = cumsum[1:]
            self._start_ends = zip(self._starts, self._ends)
        else:
            self._start_ends = start_and_ends
            self._action_spec = [end - start for (start, end) in start_and_ends]
        self._start_and_ends = list(enumerate(self._start_ends))
        self._n_subactions = len(self._start_and_ends)  # Number of sub-actions.

        self._goal_dim = goal_dim
        self._num_extra_predicted_actions = num_extra_predicted_actions

        # Decide goal conditioning style.
        if goal_dim <= 0:
            self._cbet_method = self.GOAL_SPEC.unconditional
        elif obs_dim == goal_dim:
            self._cbet_method = self.GOAL_SPEC.concat
        else:
            self._goal_encoder = nn.Linear(goal_dim, obs_dim, bias=False)
            self._cbet_method = self.GOAL_SPEC.stack

        self._gpt_model = gpt_model
        if isinstance(n_clusters, int):
            n_clusters = [n_clusters] * len(self._start_and_ends)
        assert len(n_clusters) == len(self._start_and_ends)
        # For now, we assume the number of clusters is given.
        assert all(k > 0 for k in n_clusters)
        self._K = n_clusters
        self._kmeans_fit_steps = kmeans_fit_steps
        self._clustering_algos = [
            KMeansDiscretizer(num_bins=k, kmeans_iters=kmeans_iters) for k in n_clusters
        ]
        self._current_steps = 0
        self._map_to_cbet_preds = nn.ModuleList(
            [
                MLP(
                    in_channels=gpt_model.config.output_dim,
                    hidden_channels=[(a + 1) * k],
                )
                for (a, k) in zip(self._action_spec, n_clusters)
            ]
        )
        self._collected_actions = []
        self._have_fit_kmeans = False
        # Placeholder for the cluster centers.
        generator = torch.Generator()
        generator.manual_seed(GENERATOR_SEED_FIXED)
        self._cluster_centers = nn.ParameterList(
            [
                nn.Parameter(
                    torch.randn((k, a), generator=generator, dtype=torch.float32),
                    requires_grad=False,
                )
                for (k, a) in zip(n_clusters, self._action_spec)
            ]
        )
        self._criterion = FocalLoss(gamma=gamma, reduction="none")
        self._offset_criterion = (
            nn.MSELoss(reduction="none")
            if offset_distance_metric == "L2"
            else nn.L1Loss(reduction="none")
        )
        self._offset_loss_multiplier = offset_loss_multiplier

        self._action_tokenizer = nn.ModuleList(
            [nn.Linear(a, obs_dim) for a in self._action_spec]
        )
        # Figure out the embedding tokens.
        if self._cbet_method == self.GOAL_SPEC.unconditional:
            self._goal_embedding_token = None
        else:
            self._goal_embedding_token = nn.Parameter(torch.randn(goal_dim))
        self._h, self._w = representation_height, representation_width
        self._obs_embedding_token = nn.Parameter(
            torch.randn([obs_dim, self._h, self._w])
        )
        self._action_embedding_token = nn.Parameter(
            torch.randn([self._n_subactions, obs_dim])
        )
        self._extra_action_token = nn.Parameter(
            torch.randn([self._n_subactions, obs_dim])
        )
        self._end_of_obs_token = nn.Parameter(torch.randn(obs_dim))
        self._accelerator = accelerate.Accelerator()

    def _load_from_state_dict(self, *args, **kwargs):
        # Don't fit kmeans if we are loading from a state dict.
        self._current_steps = self._kmeans_fit_steps
        self._have_fit_kmeans = True
        return super()._load_from_state_dict(*args, **kwargs)

    def get_start_and_ends(self) -> Sequence[Tuple[int, int]]:
        return self._start_and_ends

    def _tokenize_actions(self, actions: torch.Tensor) -> torch.Tensor:
        action_stack = [
            self._action_tokenizer[i](actions[..., start:end])
            for i, (start, end) in self._start_and_ends
        ]
        return torch.stack(action_stack, dim=2)

    def _detokenize_actions(
        self, tokenized_action_output: torch.Tensor
    ) -> torch.Tensor:
        action_bin_logits = []
        all_action_offsets = []
        for i, _ in self._start_and_ends:
            tokenized_action_i = tokenized_action_output[..., i, :]
            action_cbet_preds = self._map_to_cbet_preds[i](tokenized_action_i)
            action_center_logits, action_offsets = torch.split(
                action_cbet_preds,
                [self._K[i], self._K[i] * self._action_spec[i]],
                dim=-1,
            )
            action_bin_logits.append(action_center_logits)
            all_action_offsets.append(action_offsets)
        return action_bin_logits, all_action_offsets

    def _begin_epoch(self, optimizer, **kwargs):
        # log learning rate for debugging
        lr_0 = optimizer.param_groups[0]["lr"]
        lr_neg1 = optimizer.param_groups[-1]["lr"]
        return {"lr_0": lr_0, "lr_neg1": lr_neg1}

    def _calculate_cbet_preds_and_loss(
        self,
        cluster_centers: torch.Tensor,
        bin_logits: torch.Tensor,
        action_offsets: torch.Tensor,
        true_actions: torch.Tensor,
        is_padded_action_seq: Optional[torch.Tensor],
        predict_with_offset: bool = True,
        return_loss: bool = True,
        sampling_temperature: float = 1.0,
    ) -> torch.Tensor:
        bin_probs = torch.softmax(bin_logits / sampling_temperature, dim=-1)
        N, T, choices = bin_probs.shape
        # Sample from the multinomial distribution, one per row.
        sampled_centers = einops.rearrange(
            torch.multinomial(bin_probs.view(-1, choices), num_samples=1),
            "(N T) 1 -> N T 1",
            N=N,
        )
        flattened_action_offsets = einops.rearrange(
            action_offsets, "N T (K A) -> (N T) K A", K=choices
        )
        sampled_offsets = flattened_action_offsets[
            torch.arange(flattened_action_offsets.shape[0]), sampled_centers.flatten()
        ].view(N, T, -1)
        centers = cluster_centers[sampled_centers.flatten()].view(N, T, -1)
        a_hat = centers + (sampled_offsets if predict_with_offset else 0.0)
        if not return_loss:
            return a_hat, None, {}
        # We are in training, so figure out the loss for the actions.
        # First, we need to find the closest cluster center for each action.
        true_action_bins = self._find_closest_cluster(true_actions, cluster_centers)
        true_offsets = true_actions - cluster_centers[true_action_bins]
        predicted_offsets = flattened_action_offsets[
            torch.arange(flattened_action_offsets.shape[0]), true_action_bins.flatten()
        ].view(N, T, -1)
        offset_loss = self._offset_criterion(predicted_offsets, true_offsets)
        cbet_loss = self._criterion(
            einops.rearrange(bin_logits, "N T D -> (N T) D"),
            einops.rearrange(true_action_bins, "N T -> (N T)"),
        )
        if is_padded_action_seq is not None:
            cbet_loss *= ~is_padded_action_seq.view(-1)
            offset_loss *= ~is_padded_action_seq.unsqueeze(-1)
        cbet_loss, offset_loss = cbet_loss.mean(), offset_loss.mean()
        action_mse = F.mse_loss(a_hat, true_actions, reduction="none")
        action_l1 = F.l1_loss(a_hat, true_actions, reduction="none")
        norm = torch.norm(true_actions, p=2, dim=-1, keepdim=True) + 1e-9
        normalized_mse = (action_mse / norm).mean()
        loss = cbet_loss + self._offset_loss_multiplier * offset_loss
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
        if self._current_steps < self._kmeans_fit_steps and (
            action_seq is not None and padding_seq is not None
        ):
            self._current_steps += 1
            self._fit_kmeans(action_seq, padding_seq)
        return self._predict(
            obs_seq,
            goal_seq,
            action_seq,
            padding_seq,
            predict_with_offset=predict_with_offset,
        )

    def _fit_kmeans(
        self,
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
            self._collected_actions = torch.cat(self._collected_actions, dim=0)
            for i, (start, end) in self._start_and_ends:
                clustering_algo = self._clustering_algos[i]
                logging.info(f"Fitting KMeans for action {i}")
                clustering_algo.fit(
                    self._collected_actions[:, start:end].view(-1, end - start)
                )
                self._cluster_centers[i] = clustering_algo.bin_centers.float().to(
                    action_seq.device
                )
            self._have_fit_kmeans = True

    def _predict(
        self,
        obs_seq: torch.Tensor,
        goal_seq: Optional[torch.Tensor],
        action_seq: torch.Tensor,
        is_padded_action_seq: Optional[torch.Tensor],
        predict_with_offset: bool = True,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Dict[str, float]]:
        batch_size, obs_T, c, h, w = obs_seq.shape
        obs_seq_and_encoding = obs_seq + self._obs_embedding_token[None, None, ...]
        obs_seq_and_encoding = einops.rearrange(
            obs_seq_and_encoding, "N T C H W -> N T (H W) C"
        )
        assert action_seq is not None
        tokenized_action_seq = self._tokenize_actions(action_seq)
        tokenized_action_seq_and_encoding = (
            tokenized_action_seq + self._action_embedding_token[None, None, ...]
        )
        interspersed_seq = torch.cat(
            [obs_seq_and_encoding, tokenized_action_seq_and_encoding[:, :obs_T, ...]],
            dim=2,
        )  # N T (H W + A) C
        interspersed_seq_flattened = einops.rearrange(
            interspersed_seq, "N T (HWA) C -> N (T HWA) C"
        )
        extra_actions_tokenized_plus_embedding = (
            tokenized_action_seq_and_encoding[:, obs_T:, ...]
            + self._extra_action_token[None, None, ...]
        )
        extra_actions_flattened = einops.rearrange(
            extra_actions_tokenized_plus_embedding, "N T A D -> N (T A) D"
        )
        interspersed_seq_and_extra_actions = torch.cat(
            [
                interspersed_seq_flattened,
                einops.repeat(self._end_of_obs_token, "D -> N 1 D", N=batch_size),
                extra_actions_flattened,
            ],
            dim=1,
        )

        # Assume dimensions are N T D for N sequences of T timesteps with dimension D.
        if self._cbet_method == self.GOAL_SPEC.unconditional:
            gpt_input = interspersed_seq_and_extra_actions
        elif self._cbet_method == self.GOAL_SPEC.concat:
            goal_seq_encoded = goal_seq + self._goal_embedding_token[None, None, ...]
            gpt_input = torch.cat(
                [goal_seq_encoded, interspersed_seq_and_extra_actions], dim=1
            )
        elif self._cbet_method == self.GOAL_SPEC.stack:
            goal_seq_embedded = (
                goal_seq[:, None, :] + self._goal_embedding_token[None, None, ...]
            )
            goal_seq_encoded = self._goal_encoder(goal_seq_embedded)
            gpt_input = torch.cat(
                [goal_seq_encoded, interspersed_seq_and_extra_actions], dim=1
            )
        else:
            raise NotImplementedError

        gpt_output = self._gpt_model(gpt_input)
        if self._cbet_method == self.GOAL_SPEC.concat:
            # Chop off the goal encodings.
            gpt_output = gpt_output[:, goal_seq.size(1) :, :]
        elif self._cbet_method == self.GOAL_SPEC.stack:
            gpt_output = gpt_output[:, 1::]

        # Here we have a sequence of shape (N, (T*(H*W + A) + 1 + T'*A), D)
        # where T' is the number of extra actions we want to predict.
        # Separate out the original and the extra actions.
        # We have H*W + A tokens per obs, + 1 for the end of obs token.
        extra_action_output_flat = gpt_output[
            :, obs_T * (self._h * self._w + self._n_subactions) : -1, :
        ]  # N (T' A) D
        extra_action_output = einops.rearrange(
            extra_action_output_flat, "N (T A) D -> N T A D", A=self._n_subactions
        )  # N T' A D
        original_output_tokens = gpt_output[
            :, : obs_T * (self._h * self._w + self._n_subactions), :
        ]
        original_output_tokens_reshaped = einops.rearrange(
            original_output_tokens, "N (T HWA) D -> N T HWA D", T=obs_T
        )
        original_action_tokens = original_output_tokens_reshaped[
            :, :, -self._n_subactions - 1 : -1, :
        ]  # N T A D
        output_action_tokens = torch.cat(
            [original_action_tokens, extra_action_output], dim=1
        )  # N (T + T') A D
        action_bin_logits, action_offsets = self._detokenize_actions(
            output_action_tokens
        )
        # Now calculate the predicted actions and the losses.
        a_hat, loss, loss_dict = torch.zeros_like(action_seq), None, {}
        for i, (start, end) in self._start_and_ends:
            a_hat_i, loss_i, loss_dict_i = self._calculate_cbet_preds_and_loss(
                self._cluster_centers[i],
                action_bin_logits[i],
                action_offsets[i],
                action_seq[..., start:end],
                # TODO make is_padded_action_seq more flexible,
                # For example, in reality part of the action could be "padded"
                is_padded_action_seq,
                predict_with_offset=predict_with_offset,
                return_loss=True,
                sampling_temperature=self.sampling_temperature,
            )
            a_hat[..., start:end] = a_hat_i
            if i == 0:
                # a_hat = a_hat_i
                loss = loss_i
                for k, v in loss_dict_i.items():
                    loss_dict[k] = v
            else:
                # a_hat = torch.cat([a_hat, a_hat_i], dim=-1)
                loss += loss_i
                for k, v in loss_dict_i.items():
                    loss_dict[k] += v
            for k, v in loss_dict_i.items():
                loss_dict[f"{k}_{i}"] = v
        return a_hat, loss, loss_dict

    def _find_closest_cluster(
        self, action_seq: torch.Tensor, cluster_centers: torch.Tensor
    ) -> torch.Tensor:
        N, T, _ = action_seq.shape
        flattened_actions = einops.rearrange(action_seq, "N T A -> (N T) A")
        cluster_center_distance = torch.sum(
            (flattened_actions[:, None, :] - cluster_centers[None, :, :]) ** 2,
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
        optimizer.add_param_group(
            {
                "params": chain(
                    self._map_to_cbet_preds.parameters(),
                    self._action_tokenizer.parameters(),
                )
            }
        )
        optimizer.add_param_group(
            {
                "params": [
                    self._obs_embedding_token,
                    self._goal_embedding_token,
                    self._action_embedding_token,
                    self._extra_action_token,
                    self._end_of_obs_token,
                ],
                "weight_decay": 0.0,
            }
        )
        return optimizer
