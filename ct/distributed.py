"""Minimal torch.distributed helpers (works with and without torchrun)."""
import os

import torch
import torch.distributed as dist


def setup_distributed():
    """Initialize the process group if launched by torchrun.

    Returns (rank, world_size, local_rank, device).
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                                init_method="env://", world_size=world_size, rank=rank)
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        if torch.cuda.is_available():
            torch.cuda.set_device(device)
    else:
        rank, world_size, local_rank = 0, 1, 0
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def barrier():
    if is_distributed():
        dist.barrier()


def cleanup():
    if is_distributed():
        dist.destroy_process_group()
