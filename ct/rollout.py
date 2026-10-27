"""Autoregressive rollout of a full trajectory."""
import os
from contextlib import nullcontext
from functools import lru_cache

import numpy as np
import torch
from omegaconf import OmegaConf

from ct.ccd import CCDParams, CCDPostProcessor
from ct.data import build_adjacency, load_sim

DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def autocast_context(dtype: torch.dtype, device):
    if dtype == torch.float32 or torch.device(device).type != "cuda":
        return nullcontext()
    return torch.autocast("cuda", dtype=dtype)


@lru_cache(maxsize=None)
def _load_case_config(path: str) -> dict:
    if not os.path.isfile(path):
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(repo_root, path)
    return OmegaConf.to_container(OmegaConf.load(path)) or {}


def ccd_params_for(cfg, sim_path: str) -> CCDParams:
    """CCD post-processing parameters for one trajectory: ``ccd_postprocess``
    overridden by the entry ``<category>/<sim_id>`` of ``ccd_postprocess.case_config``."""
    params = OmegaConf.to_container(cfg.ccd_postprocess)
    if cfg.ccd_postprocess.case_config:
        case_id = "/".join(os.path.normpath(sim_path)[:-len(".npz")].split(os.sep)[-2:])
        params.update(_load_case_config(str(cfg.ccd_postprocess.case_config)).get(case_id) or {})
    return CCDParams.from_config(params)


@torch.no_grad()
def rollout(model, sim_path: str, cfg, step_size=None) -> np.ndarray:
    """Roll the model out over a whole trajectory, starting from the ground-truth
    first ``block_size`` frames and feeding back its own predictions.

    Args:
        model: the model (optionally DDP-wrapped)
        sim_path: path to a ``sim_XXXXX.npz`` file (initial state + collider motion)
        cfg: full config
        step_size: number of predicted frames kept per model call (<= block_size)

    Returns:
        (T, N_v, 3) predicted cloth positions; the first ``block_size`` frames are the
        initial state.
    """
    base = model.module if hasattr(model, "module") else model
    base.eval()
    device = next(base.parameters()).device
    block = int(cfg.model.block_size)
    step_size = block if step_size is None else int(step_size)
    assert 0 < step_size <= block, f"step_size {step_size} must be in (0, block_size={block}]"
    dt = float(cfg.data.dt)
    vel_scale = float(cfg.data.velocity_scale)
    dtype = DTYPES[cfg.runtime.dtype]

    sim = load_sim(sim_path, dt)
    f32 = torch.float32
    init = torch.from_numpy(sim["traj"][:block]).unsqueeze(0).to(f32).to(device)        # (1, block, N_v, 3)
    init_vel = (torch.from_numpy(sim["traj_vel"][:block]).unsqueeze(0).to(f32) * vel_scale).to(device)
    col_seq = torch.from_numpy(sim["collision"]).unsqueeze(0).to(f32).to(device)        # (1, T, N_tris, 9)
    col_vel_seq = (torch.from_numpy(sim["collision_vel"]).unsqueeze(0).to(f32) * vel_scale).to(device)
    rest_pos = torch.from_numpy(sim["rest_pos"]).to(f32).unsqueeze(0).to(device)
    edges = torch.from_numpy(sim["edges"]).to(torch.long)
    T, N_v = col_seq.shape[1], init.shape[2]
    adj = build_adjacency(edges, N_v).unsqueeze(0).to(device)

    ccd = ccd_params = None
    if cfg.ccd_postprocess.enabled:
        ccd_params = ccd_params_for(cfg, sim_path)
        if ccd_params.enabled:
            ccd = CCDPostProcessor(torch.from_numpy(sim["triangles"]).to(device), edges.to(device),
                                   torch.from_numpy(sim["collision_triangles"]).to(device),
                                   torch.from_numpy(sim["collision_edges"]).to(device), ccd_params)
    col_verts_seq = torch.from_numpy(sim["collision_vertices"]).to(f32).to(device)      # (T, N_c, 3)

    # The history window holds `block` frames and starts as the initial state.
    history, history_vel = init.clone(), init_vel.clone()
    prev_out = history[0, -1]  # last output frame (CCD-corrected if enabled)
    pred_traj = np.zeros((T, N_v, 3), dtype=np.float32)
    pred_traj[:block] = sim["traj"][:block]

    frame = block  # index of the first frame to predict
    while frame + block <= T:
        # Collider input is shifted one block ahead of the cloth (lookahead).
        with autocast_context(dtype, device):
            pred = base(history, col_seq[:, frame:frame + block], history_vel,
                        col_vel_seq[:, frame:frame + block], rest_pos, adj_matrix=adj)
        raw_block = pred[:, -block:].float()  # (1, block, N_v, 3)
        out_block = raw_block
        if ccd is not None:
            # With feedback the corrected frames become the model input; otherwise
            # the model rolls out on its raw predictions and only the output is corrected.
            out_block = raw_block.clone()
            for i in range(block):
                fixed, n_it, n_hits = ccd(prev_out, raw_block[0, i], col_verts_seq[frame - 1 + i],
                                          col_verts_seq[frame + i])
                out_block[0, i] = fixed
                prev_out = fixed
                if cfg.ccd_postprocess.verbose:
                    print(f"  [CCD] frame {frame + i}: {n_hits} contacts, {n_it} iterations")

        keep = out_block[:, :step_size]
        prev_out = keep[0, -1]
        pred_traj[frame:frame + step_size] = keep[0].cpu().numpy()

        # Slide the window; velocities of predicted frames are finite differences.
        feed = raw_block[:, :step_size] if (ccd is not None and not ccd_params.feedback) else keep
        new_vel = torch.diff(torch.cat([history[:, -1:], feed], dim=1), dim=1) / dt * vel_scale
        history = torch.cat([history[:, step_size:], feed], dim=1)
        history_vel = torch.cat([history_vel[:, step_size:], new_vel], dim=1)
        frame += step_size
    return pred_traj
