import os
import wandb
from omegaconf import OmegaConf


def init_wandb(cfg):
    if cfg.wandb.api_key:
        os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
    wandb.login()
    wandb.init(project=cfg.wandb.project, config=OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
