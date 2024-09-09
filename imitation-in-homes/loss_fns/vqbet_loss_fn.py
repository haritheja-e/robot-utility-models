from typing import Optional

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange

from loss_fns.abstract_loss_fn import AbstractLossFn
from models.bet import GPT
from models.bet.vqvae.vqvae import VqVae
from models.bet.vqbet import VQBehaviorTransformer


class VQBeTLossFn(AbstractLossFn):
    def __init__(
        self,
        tokenized_bet: bool,
        action_dim: int,
        xyz_only: bool,
        vqvae_load_dir: str,
        gpt_model: GPT,
        goal_dim: int = 0,
        mask_last_min: int = 0,
        mask_last_max: int = 0,
        learned_mask: bool = True,
        use_depth: bool = False,
        model: Optional[torch.nn.Module] = None,
        action_sequence_length: int = 1,
        vqvae_n_latent_dims: int = 512,
        vqvae_n_embed: int = 16,
        vqvae_groups: int = 2,
        obs_cond: bool = False,
        offset_loss_multiplier: float = 100.0,
        secondary_code_multiplier: float = 0.5,
        gamma: float = 2.0,
        obs_window_size: int = 10,
        sequentially_select: bool = False,
        temperature: float = 1.0,
        device: str = "cuda",
    ):
        super().__init__(model)
        assert mask_last_max >= mask_last_min
        assert mask_last_min >= 0
        obs_dim = model.feature_dim if not use_depth else model.feature_dim * 2
        gpt_input_dim = gpt_model.config.input_dim
        if use_depth:
            self._depth_net = DepthNet(model.feature_dim)
        self._use_depth = use_depth

        # TODO (mahi): currently, we are casting everything to a concat style goal
        #  but we should be able to handle different types of goals like concat or stack
        self._use_goals = goal_dim > 0
        if self._use_goals and goal_dim != gpt_input_dim:
            self._goal_adapter = nn.Sequential(
                nn.Linear(goal_dim, gpt_input_dim, bias=False),
                Rearrange("b g -> b 1 g"),
            )
            goal_dim = gpt_input_dim
        else:
            self._goal_adapter = Rearrange("b g -> b 1 g")
        self._true_action_dim = action_dim
        action_dim = 3 if xyz_only else action_dim
        self._rvq = VqVae(
            obs_dim=gpt_input_dim,
            input_dim_h=action_sequence_length,
            input_dim_w=action_dim,
            n_latent_dims=vqvae_n_latent_dims,
            vqvae_n_embed=vqvae_n_embed,
            vqvae_groups=vqvae_groups,
            device=device,
            eval=True,
            enc_loss_type="through_vqlayer",
            obs_cond=obs_cond,
            load_dir=vqvae_load_dir,
        )

        for param in self._rvq.parameters():
            param.requires_grad = False
        self._vqbet = VQBehaviorTransformer(
            obs_dim=gpt_input_dim,
            act_dim=action_dim,
            goal_dim=goal_dim,
            gpt_model=gpt_model,
            vqvae_model=self._rvq,
            offset_loss_multiplier=offset_loss_multiplier,
            secondary_code_multiplier=secondary_code_multiplier,
            gamma=gamma,
            obs_window_size=obs_window_size,
            act_window_size=action_sequence_length,
            sequentially_select=sequentially_select,
            temperature=temperature,
            device=device,
        )

        self._obs_mask_token = (
            nn.Parameter(torch.ones(gpt_input_dim), requires_grad=False)
            if not learned_mask or mask_last_max == 0
            else nn.Parameter(torch.randn(gpt_input_dim), requires_grad=True)
        )
        self._obs_adapter = nn.Linear(obs_dim, gpt_input_dim, bias=False)
        # self._adapt_obs = nn.Linear(obs_dim, gpt_input_dim, bias=False)
        self._mask_last = (mask_last_min, mask_last_max)
        self._action_dim = action_dim

        self._adapt_obs = (
            self._adapt_obs_linear if not tokenized_bet else self._adapt_obs_tokenized
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
        return self._vqbet._begin_epoch(*args, **kwargs)

    def forward(self, data, output, *args, **kwargs):
        # TODO Mahi fix the order of depth and goals.
        if self._use_goals:
            _, goals, *_, padding, actions = data
            if self._use_depth:
                _, goals, *_, depths, padding, actions = data
                output = torch.cat([output, self._depth_net(depths)], dim=-1)
            goals = self._goal_adapter(goals)
        else:
            *_, padding, actions = data
            goals = None
            if self._use_depth:
                *_, depths, padding, actions = data
                output = torch.cat([output, self._depth_net(depths)], dim=-1)
        adapted_obs = self._adapt_obs(output)
        if "second_half" in kwargs:
            second_half = kwargs["second_half"]
        else:
            second_half = False
        action_seq = actions[..., : self._action_dim].contiguous()
        _, loss, loss_dict = self._vqbet(
            adapted_obs,
            goal_seq=goals,
            action_seq=action_seq,
            second_half=second_half,
        )
        return loss, loss_dict

    @torch.no_grad()
    def _step(self, data, output, *args, **kwargs):
        if self._use_depth:
            *_, depths, padding, actions = data
            output = torch.cat([output, self._depth_net(depths)], dim=-1)
        goals = data[1] if self._use_goals else None
        adapted_obs = self._adapt_obs(output)
        goals = self._goal_adapter(goals) if self._use_goals else None
        a_hat, _, _ = self._vqbet(
            adapted_obs,
            goal_seq=goals,
            action_seq=None,
        )
        if a_hat.shape[2] != self._true_action_dim:
            # append n zeros and a 1 to the end of y_hat
            a_hat = torch.cat(
                [
                    a_hat,
                    torch.zeros(
                        a_hat.shape[0],
                        a_hat.shape[1],
                        self._true_action_dim - a_hat.shape[2],
                    ).to(a_hat.device),
                ],
                dim=2,
            )
            a_hat[:, :, -1] = 1.0
        # Finally, return the final action prediction only.
        return a_hat[-1, -1, :], {}
