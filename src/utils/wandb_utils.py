import os
import wandb

def init_wandb(cfg):
    os.environ["WANDB_API_KEY"] = cfg.wandb.api_key
    wandb.login()
    wandb.init(project=cfg.wandb.project, config=cfg)
