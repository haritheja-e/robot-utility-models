import torch
import torch.nn as nn
import einops
from models.bet.vqvae.residual_vq import ResidualVQ
from models.bet.vqvae.vqvae_utils import get_tensor, weights_init_encoder


class EncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=128,
        layer_num=1,
        last_activation=None,
    ):
        super(EncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x):
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


class CondiitonalEncoderMLP(nn.Module):
    def __init__(
        self,
        input_dim,
        output_dim=16,
        hidden_dim=512,
        layer_num=2,
        last_activation=None,
        obs_dim=None,
    ):
        super(CondiitonalEncoderMLP, self).__init__()
        layers = []

        layers.append(nn.Linear(input_dim + obs_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(layer_num):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())

        self.encoder = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden_dim, output_dim)

        if last_activation is not None:
            self.last_layer = last_activation
        else:
            self.last_layer = None
        self.apply(weights_init_encoder)

    def forward(self, x, obs):
        x = torch.cat((x, obs), dim=1)
        h = self.encoder(x)
        state = self.fc(h)
        if self.last_layer:
            state = self.last_layer(state)
        return state


class VqVae(nn.Module):
    def __init__(
        self,
        obs_dim=60,
        input_dim_h=10,  # length of action chunk
        input_dim_w=9,  # action dim
        n_latent_dims=512,
        vqvae_n_embed=32,
        vqvae_groups=4,
        eval=True,
        device="cuda",
        load_dir=None,
        enc_loss_type="skip_vqlayer",
        obs_cond=False,
        encoder_loss_multiplier=1.0,
        act_scale=1.0,
    ):
        super(VqVae, self).__init__()
        self.n_latent_dims = n_latent_dims  # 64
        self.input_dim_h = input_dim_h
        self.input_dim_w = input_dim_w
        self.rep_dim = self.n_latent_dims
        self.vqvae_n_embed = vqvae_n_embed  # 120
        self.vqvae_lr = 1e-3
        self.vqvae_groups = vqvae_groups
        self.device = device
        self.enc_loss_type = enc_loss_type
        self.obs_cond = obs_cond
        self.encoder_loss_multiplier = encoder_loss_multiplier
        self.act_scale = act_scale

        discrete_cfg = {"groups": self.vqvae_groups, "n_embed": self.vqvae_n_embed}
        self.vq_layer = ResidualVQ(
            dim=self.n_latent_dims,
            num_quantizers=discrete_cfg["groups"],
            codebook_size=self.vqvae_n_embed,
            eval=eval,
        ).to(self.device)
        self.embedding_dim = self.n_latent_dims
        self.vq_layer.device = device

        if self.input_dim_h == 1:
            if self.obs_cond:
                self.encoder = CondiitonalEncoderMLP(
                    input_dim=input_dim_w, output_dim=n_latent_dims, obs_dim=obs_dim
                ).to(self.device)
                self.decoder = CondiitonalEncoderMLP(
                    input_dim=n_latent_dims, output_dim=input_dim_w, obs_dim=obs_dim
                ).to(self.device)
            else:
                self.encoder = EncoderMLP(
                    input_dim=input_dim_w, output_dim=n_latent_dims
                ).to(self.device)
                self.decoder = EncoderMLP(
                    input_dim=n_latent_dims, output_dim=input_dim_w
                ).to(self.device)
        else:
            if self.obs_cond:
                self.encoder = CondiitonalEncoderMLP(
                    input_dim=input_dim_w * self.input_dim_h,
                    output_dim=n_latent_dims,
                    obs_dim=obs_dim,
                ).to(self.device)
                self.decoder = CondiitonalEncoderMLP(
                    input_dim=n_latent_dims,
                    output_dim=input_dim_w * self.input_dim_h,
                    obs_dim=obs_dim,
                ).to(self.device)
            else:
                self.encoder = EncoderMLP(
                    input_dim=input_dim_w * self.input_dim_h, output_dim=n_latent_dims
                ).to(self.device)
                self.decoder = EncoderMLP(
                    input_dim=n_latent_dims, output_dim=input_dim_w * self.input_dim_h
                ).to(self.device)

        # params = list(self.encoder.parameters()) + list(self.decoder.parameters()) + list(self.vq_layer.parameters())
        # self.vqvae_optimizer = torch.optim.Adam(params, lr=self.vqvae_lr, weight_decay=0.0001)

        if load_dir is not None:
            try:
                state_dict = torch.load(load_dir)
            except RuntimeError:
                state_dict = torch.load(load_dir, map_location=torch.device("cpu"))

            new_dict = {}

            prefix_to_remove = "_rvq."

            for key, value in state_dict["loss_fn"].items():
                if key.startswith(prefix_to_remove):
                    new_key = key[len(prefix_to_remove) :]  # Remove the prefix
                    new_dict[new_key] = value
                else:
                    new_dict[key] = value

            self.load_state_dict(new_dict, strict=False)

        if eval:
            self.vq_layer.eval()
        else:
            self.vq_layer.train()

    def draw_logits_forward(self, encoding_logits):
        z_embed = self.vq_layer.draw_logits_forward(encoding_logits)
        return z_embed

    def draw_code_forward(self, encoding_indices):
        with torch.no_grad():
            z_embed = self.vq_layer.get_codes_from_indices(encoding_indices)
            z_embed = z_embed.sum(dim=0)
        return z_embed

    def get_action_from_latent(self, latent, obs=None):
        if self.obs_cond:
            output = self.decoder(latent, obs[:, -1]) * self.act_scale
        else:
            output = self.decoder(latent) * self.act_scale
        if self.input_dim_h == 1:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)
        else:
            return einops.rearrange(output, "N (T A) -> N T A", A=self.input_dim_w)

    def preprocess(self, state):
        if not torch.is_tensor(state):
            state = get_tensor(state, self.device)
        if self.input_dim_h == 1:
            state = state.squeeze(-2)  # state.squeeze(-1)
        else:
            state = einops.rearrange(state, "N T A -> N (T A)")
        return state.to(self.device)

    def get_code(self, state, obs=None, required_recon=False):
        state = state / self.act_scale
        state = self.preprocess(state)
        with torch.no_grad():
            if self.obs_cond:
                state_rep = self.encoder(state, obs[:, -1])
            else:
                state_rep = self.encoder(state)
            state_rep_shape = state_rep.shape[:-1]
            state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
            state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
            state_vq = state_rep_flat.view(*state_rep_shape, -1)
            vq_code = vq_code.view(*state_rep_shape, -1)
            vq_loss_state = torch.sum(vq_loss_state)
            if required_recon:
                if self.obs_cond:
                    recon_state = self.decoder(state_vq, obs[:, -1]) * self.act_scale
                    recon_state_ae = (
                        self.decoder(state_rep, obs[:, -1]) * self.act_scale
                    )
                else:
                    recon_state = self.decoder(state_vq) * self.act_scale
                    recon_state_ae = self.decoder(state_rep) * self.act_scale
                if self.input_dim_h == 1:
                    return state_vq, vq_code, recon_state, recon_state_ae
                else:
                    return (
                        state_vq,
                        vq_code,
                        torch.swapaxes(recon_state, -2, -1),
                        torch.swapaxes(recon_state_ae, -2, -1),
                    )
            else:
                return state_vq, vq_code

    def vqvae_update(self, state, obs=None):
        state = state / self.act_scale
        state = self.preprocess(state)
        if self.obs_cond:
            state_rep = self.encoder(state, obs[:, -1])
        else:
            state_rep = self.encoder(state)
        state_rep_shape = state_rep.shape[:-1]
        state_rep_flat = state_rep.view(state_rep.size(0), -1, state_rep.size(1))
        state_rep_flat, vq_code, vq_loss_state = self.vq_layer(state_rep_flat)
        state_vq = state_rep_flat.view(*state_rep_shape, -1)
        vq_code = vq_code.view(*state_rep_shape, -1)
        vq_loss_state = torch.sum(vq_loss_state)

        if self.obs_cond:
            dec_out = self.decoder(state_vq, obs[:, -1])
        else:
            dec_out = self.decoder(state_vq)
        encoder_loss = (state - dec_out).abs().mean()

        rep_loss = encoder_loss * self.encoder_loss_multiplier + (vq_loss_state * 5)

        # self.vqvae_optimizer.zero_grad()
        # rep_loss.backward()
        # self.vqvae_optimizer.step()
        vqvae_recon_loss = torch.nn.MSELoss()(state, dec_out)
        loss_dict = {
            "rep_loss": rep_loss.detach().cpu().item(),
            "vq_loss_state": vq_loss_state.detach().cpu().item(),
            "vqvae_recon_loss_l1": encoder_loss.detach().cpu().item(),
            "vqvae_recon_loss_l2": vqvae_recon_loss.detach().cpu().item(),
            "n_different_codes": len(torch.unique(vq_code)),
            "n_different_combinations": len(torch.unique(vq_code, dim=0)),
        }
        return rep_loss, loss_dict

    def configure_optimizers(self, weight_decay, learning_rate, betas):
        params = (
            list(self.encoder.parameters())
            + list(self.decoder.parameters())
            + list(self.vq_layer.parameters())
        )
        optimizer = torch.optim.AdamW(
            params,
            weight_decay=weight_decay,
            learning_rate=learning_rate,
            betas=betas,
        )
        return optimizer

    def _begin_epoch(self, optimizer, **kwargs):
        # log codebook usage rate for debugging
        # lr_0 = optimizer.param_groups[0]["lr"]
        # lr_neg1 = optimizer.param_groups[-1]["lr"]
        # return {"lr_0": lr_0, "lr_neg1": lr_neg1}
        return None

    # def state_dict(self):
    #     return {'encoder': self.encoder.state_dict(),
    #             'decoder': self.decoder.state_dict(),
    #             'vq_embedding': self.vq_layer.state_dict()}

    # def load_state_dict(self, state_dict):
    #     self.encoder.load_state_dict(state_dict['encoder'])
    #     self.decoder.load_state_dict(state_dict['decoder'])
    #     self.vq_layer.load_state_dict(state_dict['vq_embedding'])
    #     self.vq_layer.eval()
