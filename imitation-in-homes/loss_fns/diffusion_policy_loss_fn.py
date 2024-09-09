from typing import Optional, Sequence, Tuple

import einops
import torch
import torch.nn as nn
from itertools import chain

from loss_fns.abstract_loss_fn import AbstractLossFn
from models.policies.diffusion_policy import DiffusionPolicy

class DiffusionPolicyLossFn(AbstractLossFn):
    def __init__(
        self,
        tokenized_bet: bool,
        action_dim: int,
        obs_dim: int,
        xyz_only: bool,
        mask_last_min: int = 0,
        mask_last_max: int = 0,
        learned_mask: bool = True,
        use_depth: bool = False,
        model: Optional[torch.nn.Module] = None,
        obs_window_size: int = 10,
        action_sequence_length: int = 1,
        data_act_scale: float = 1.0,
        data_obs_scale: float = 1.0,
        policy_type: str = 'cnn',
        device: str = 'cuda', 
    ):
        super().__init__(model)
        assert mask_last_max >= mask_last_min
        assert mask_last_min >= 0
        obs_dim = model.feature_dim if not use_depth else model.feature_dim * 2
        if use_depth:
            self._depth_net = DepthNet(model.feature_dim)
        self._use_depth = use_depth
        self._true_action_dim = action_dim
        action_dim = 3 if xyz_only else action_dim

        self._diffusionpolicy = DiffusionPolicy(
            obs_dim= obs_dim,
            act_dim=action_dim,
            obs_horizon = obs_window_size,
            pred_horizon = (obs_window_size + action_sequence_length - 1),
            action_horizon = action_sequence_length,
            data_act_scale = data_act_scale,
            data_obs_scale = data_obs_scale,
            policy_type = policy_type,
            device = device, 
        )


        self._obs_mask_token = (
            nn.Parameter(torch.ones(obs_dim), requires_grad=False)
            if not learned_mask or mask_last_max == 0
            else nn.Parameter(torch.randn(obs_dim), requires_grad=True)
        )
        self._obs_adapter = nn.Linear(obs_dim, obs_dim, bias=False)
        self._adapt_obs = (
            self._adapt_obs_linear if not tokenized_bet else self._adapt_obs_tokenized
        )
        self._mask_last = (mask_last_min, mask_last_max)
        self._action_dim = action_dim

        self.step = self._step if not tokenized_bet else self._step_tokenized

        self._seen_action_stack = []
        self._start_and_ends = None

    def ema_step(self):
        self._diffusionpolicy.ema_step()

    def _adapt_obs_tokenized(self, obs):
        return einops.rearrange(
            self._obs_adapter(einops.rearrange(obs, "... c h w -> ... h w c")),
            "... h w c -> ... c h w",
        )

    def _adapt_obs_linear(self, obs):
        return self._obs_adapter(obs)

    def _begin_epoch(self, *args, **kwargs):
        return self._diffusionpolicy._begin_epoch(*args, **kwargs)

    def forward(self, data, output, eval = False, *args, **kwargs):
        *_, padding, actions = data
        if self._use_depth:
            *_, depths, padding, actions = data
            output = torch.cat([output, self._depth_net(depths)], dim=-1)
        adapted_obs = self._adapt_obs(output)
        if actions is not None:
            action_seq = actions[..., : self._action_dim].contiguous()
        else:
            action_seq = None
        _, loss, loss_dict = self._diffusionpolicy(
            adapted_obs,
            action_seq=action_seq,
            eval = eval
        )
        return loss, loss_dict


    @torch.no_grad()
    def _step(self, data, output, *args, **kwargs):
        if self._use_depth:
            _, depths, *_ = data
            output = torch.cat([output, self._depth_net(depths)], dim=-1)
        adapted_obs = self._adapt_obs(output)
        a_hat, _, _ = self._diffusionpolicy(
            adapted_obs,
            action_seq=None,
            eval=True,
        )
        
        return a_hat[0], {}

        # return a_hat, {}
