from typing import Optional, Sequence, Tuple

import einops
import torch
import torch.nn as nn

from loss_fns.abstract_loss_fn import AbstractLossFn
from models.bet import GPT, BehaviorTransformer, TokenizedBehaviorTransformer
from models.bet.vqvae.vqvae import VqVae


class RVQLossFn(AbstractLossFn):
    def __init__(
        self,
        tokenized_bet: bool,
        action_dim: int,
        xyz_only: bool,
        mask_last_min: int = 0,
        mask_last_max: int = 0,
        gpt_input_dim: int = 0,
        learned_mask: bool = True,
        model: Optional[torch.nn.Module] = None,
        predict_with_offsets: bool = True,
        sampling_temperature: float = 1.0,
        action_sequence_length: int = 1,
        vqvae_n_latent_dims: int = 512,
        vqvae_n_embed: int = 16,
        vqvae_groups: int = 2,
        obs_cond: bool = False,
    ):
        super().__init__(model)
        assert mask_last_max >= mask_last_min
        assert mask_last_min >= 0
        obs_dim = model.feature_dim
        self._true_action_dim = action_dim
        action_dim = 3 if xyz_only else action_dim
        self.obs_cond = obs_cond
        self._model = model
        self._rvq = VqVae(
            obs_dim=gpt_input_dim,
            input_dim_h=action_sequence_length,
            input_dim_w=action_dim,
            n_latent_dims=vqvae_n_latent_dims,
            vqvae_n_embed=vqvae_n_embed,
            vqvae_groups=vqvae_groups,
            eval=False,
            enc_loss_type="through_vqlayer",
            obs_cond=obs_cond,
        )

        if self.obs_cond:
            self._obs_mask_token = (
                nn.Parameter(torch.ones(gpt_input_dim), requires_grad=False)
                if not learned_mask or mask_last_max == 0
                else nn.Parameter(torch.randn(gpt_input_dim), requires_grad=True)
            )
            self._obs_adapter = nn.Linear(obs_dim, gpt_input_dim, bias=False)
            self._adapt_obs = (
                self._adapt_obs_linear
                if not tokenized_bet
                else self._adapt_obs_tokenized
            )
        self._mask_last = (mask_last_min, mask_last_max)
        self._action_dim = action_dim

        self.step = self._step if not tokenized_bet else self._step_tokenized

        self._seen_action_stack = []
        self._start_and_ends = None

    def _adapt_obs_tokenized(self, obs):
        return einops.rearrange(
            self._obs_adapter(einops.rearrange(obs, "... c h w -> ... h w c")),
            "... h w c -> ... c h w",
        )

    def _adapt_obs_linear(self, obs):
        return self._obs_adapter(obs)

    def _begin_epoch(self, *args, **kwargs):
        return self._rvq._begin_epoch(*args, **kwargs)

    def forward(self, data, output, *args, **kwargs):
        *_, padding, actions = data
        action_seq = actions[..., : self._action_dim].contiguous()
        if self.obs_cond:
            adapted_obs = self._adapt_obs(output)
            loss, loss_dict = self._rvq.vqvae_update(action_seq, adapted_obs)
        else:
            loss, loss_dict = self._rvq.vqvae_update(action_seq, None)
        return loss, loss_dict

    @torch.no_grad()
    def _step(self, data, output, *args, **kwargs):
        pass

    @torch.no_grad()
    def _step_tokenized(self, data, output, *args, **kwargs):
        pass
