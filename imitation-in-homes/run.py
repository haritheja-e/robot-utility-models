from typing import Tuple

import cv2
import hydra
import torch
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader
import os
import gdown

import wandb
from utils.trajectory_vis import visualize_trajectory

TASK_GDRIVE_ID = {
    "vqbet": {
        "door_opening": "17fWsurnkp1-UtFg9BEK0t3nOWryrhh7g",
        "drawer_opening": "1B13lIdFeqXGnxAPUtsrtgKe5K5bI-pS0",
        "reorientation": "1Shhs8rMA8EIF46N7_-D8yES02kg_hK2w",
        "bag_pick_up": "1LpmdIQ7-pV7BIqiTFWzs70nK3JiMyRwz",
        "tissue_pick_up": "1tw03YyFUBM0nVEG_DRDVvvDed3ftU3dH",
    },
    "diffusion": {
        "door_opening": "1G8ZuhXnfDrZiugba9TktMPGc65K5NX0j",
        "drawer_opening": "1ETnWSjddHwsdp9xnWi19UGduukYvpy7m",
        "reorientation": "1ClyHWjhM9RpT18XB5DG_mupVPl-81E6N",
        "bag_pick_up": "1pqFwzXxV7Gm80r8gbtiHEa7dD9Z8GSAX",
        "tissue_pick_up": "1HPT1Vz82DAANn0B3NYe5N1iDxGlRVjhQ",
    }
}

class WrapperPolicy(nn.Module):
    def __init__(self, model, loss_fn):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn

    def step(self, data, *args, **kwargs):
        model_out = self.model(data)
        return self.loss_fn.step(data, model_out)

    def reset(self):
        pass

def _init_run(cfg: OmegaConf):
    dict_cfg = OmegaConf.to_container(cfg, resolve=True)
    wandb.init(
        project=cfg.wandb.project,
        mode="disabled",
        entity=cfg.wandb.entity,
        config=dict_cfg,
    )
    return dict_cfg

def get_model_weight_pth(task_name, policy_class):
    if task_name not in TASK_GDRIVE_ID[policy_class]:
        raise ValueError(f"Task \'{task_name}\' is invalid. Please choose from {TASK_GDRIVE_ID[policy_class].keys()}.")
    task_dir = f"checkpoints/{policy_class}/{task_name}"
    if not os.path.exists(f"{task_dir}/checkpoint.pt"):
        os.makedirs(task_dir)
        gdrive_id = TASK_GDRIVE_ID[policy_class][task_name]
        url = f'https://drive.google.com/uc?id={gdrive_id}'
        output_path = f"{task_dir}/checkpoint.pt"
        gdown.download(url, output_path, quiet=False)
    
    return f"{task_dir}/checkpoint.pt"

def _init_model_loss(cfg):
    model = hydra.utils.instantiate(cfg.model)
    model = model.to(cfg.device)

    model_weight_pth = cfg.get("model_weight_pth")
    if model_weight_pth is None:
        if "vqbet" in cfg.loss_fn._target_:
            policy_class = "vqbet"
        elif "diffusion" in cfg.loss_fn._target_:
            policy_class = "diffusion"
        else:
            raise ValueError(f"Policy class in loss_fn not found.")
        model_weight_pth = get_model_weight_pth(task_name=cfg.task, policy_class=policy_class)
    checkpoint = torch.load(model_weight_pth, map_location=cfg.device)

    model.load_state_dict(checkpoint["model"])
    loss_fn = hydra.utils.instantiate(cfg.loss_fn, model=model)
    loss_fn.load_state_dict(checkpoint["loss_fn"])
    loss_fn = loss_fn.to(cfg.device)
    
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    loss_parameters = sum(p.numel() for p in loss_fn.parameters() if p.requires_grad)
    # print model params in millions with %.2f
    print(f"Model parameters: {model_parameters / 1e6:.2f}M")
    print(f"Loss parameters: {loss_parameters / 1e6:.2f}M")
        
    policy = WrapperPolicy(model, loss_fn)
    return policy


def run(cfg: OmegaConf, init_model=_init_model_loss):
    model = init_model(cfg)
    if cfg["run_offline"] is True:
        test_dataset = hydra.utils.instantiate(cfg.dataset.test)
        visualize_trajectory(
            model,
            test_dataset,
            cfg["device"],
            cfg["image_buffer_size"],
            goal_conditional=cfg["goal_conditional"],
        )

    else:
        # Lazy loading so we can run offline eval without the robot set up.
        from robot.controller import Controller

        dict_cfg = _init_run(cfg)
        controller = Controller(cfg=dict_cfg)
        controller.setup_model(model)
        controller.run()


@hydra.main(config_path="configs", config_name="run_vqbet", version_base="1.2")
def main(cfg: OmegaConf):
    run(cfg)

if __name__ == "__main__":
    main()
