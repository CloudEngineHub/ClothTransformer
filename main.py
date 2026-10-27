#!/usr/bin/env python3
"""ClothTransformer training and inference entry point.

    # train (single GPU; see scripts/run.sh for multi-GPU / Slurm)
    python main.py
    # roll out a checkpoint on the test split and export .obj sequences
    python main.py infer.ckpt=checkpoints/ct-no-ccd-preview.pt

Any config field can be overridden with OmegaConf dot-notation (key=value);
``--config path.yaml`` replaces configs/default.yaml as the base config.
"""
import os
import random
import sys

import torch
from omegaconf import OmegaConf

from ct.data import build_splits
from ct.distributed import cleanup, setup_distributed
from ct.models import build_model

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def load_config(argv):
    base = os.path.join(REPO_ROOT, "configs", "default.yaml")
    overrides = []
    i = 0
    while i < len(argv):
        if argv[i] == "--config":
            base = argv[i + 1]
            i += 2
            continue
        if "=" not in argv[i]:
            raise SystemExit(f"Unrecognized argument {argv[i]!r}; expected key=value or --config FILE")
        overrides.append(argv[i])
        i += 1
    cfg = OmegaConf.merge(OmegaConf.load(base), OmegaConf.from_dotlist(overrides))
    OmegaConf.set_readonly(cfg, True)
    return cfg


def load_weights(model, path, device):
    state = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    print(f"Loaded weights from {path}")


def main():
    cfg = load_config(sys.argv[1:])
    rank, world_size, local_rank, device = setup_distributed()
    seed = int(cfg.training.seed) + rank
    torch.manual_seed(seed)
    random.seed(seed)

    # Explicit trajectories (infer.sim / infer.file_list) do not need the full dataset split.
    explicit = cfg.infer.ckpt and (cfg.infer.sim or cfg.infer.file_list)
    splits = None if explicit else build_splits(cfg, verbose=rank == 0)
    model = build_model(cfg.model).to(device)
    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Model parameters: {n_params:,}")

    if cfg.infer.ckpt:
        from ct.inference import run_inference
        load_weights(model, cfg.infer.ckpt, device)
        run_inference(cfg, model, splits, rank, world_size)
    else:
        from ct.trainer import Trainer
        if cfg.training.init_ckpt:
            load_weights(model, cfg.training.init_ckpt, device)
        Trainer(cfg, model, splits, rank, world_size, local_rank, device).fit()
    cleanup()


if __name__ == "__main__":
    main()
