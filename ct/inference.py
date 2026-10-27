"""Inference: autoregressive rollouts of a set of trajectories, exported as .obj sequences."""
import os
import time

import numpy as np

from ct.rollout import ccd_params_for, rollout

SPLIT_PRESETS = {
    "train": ["train"],
    "val": ["val"],
    "test": ["test"],
    "train_val": ["train", "val"],
    "all": ["train", "val", "test"],
}


def write_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w") as f:
        np.savetxt(f, verts, fmt="v %.8f %.8f %.8f")
        np.savetxt(f, faces + 1, fmt="f %d %d %d")


def export_sequences(sim_path: str, pred_traj: np.ndarray, out_dir: str, pred_name: str = "pred"):
    """Write frame-aligned .obj sequences of prediction, ground truth and collider:

        <out_dir>/<category>/<sim_id>/{pred,gt,collision}/frame_XXXX.obj

    The first ``block_size`` frames of ``pred`` are the ground-truth initial state.
    """
    with np.load(sim_path) as data:
        faces = data["triangles"]
        gt = data["traj"][:, :, :3]
        col_verts = data["collision_vertices"]
        col_faces = data["collision_triangles"]
    category = os.path.basename(os.path.dirname(sim_path))
    sim_id = os.path.basename(sim_path)[:-len(".npz")]
    base = os.path.join(out_dir, category, sim_id)
    n = min(pred_traj.shape[0], gt.shape[0], col_verts.shape[0])
    for name, verts, f in ((pred_name, pred_traj, faces), ("gt", gt, faces), ("collision", col_verts, col_faces)):
        os.makedirs(os.path.join(base, name), exist_ok=True)
        for i in range(n):
            write_obj(os.path.join(base, name, f"frame_{i:04d}.obj"), verts[i], f)
    return base


def select_files(cfg, splits):
    """Trajectories to run: ``infer.sim`` or ``infer.file_list`` if set, else the configured split(s)."""
    c = cfg.infer
    if c.sim:
        files = [os.path.join(cfg.data.dir, c.sim if c.sim.endswith(".npz") else c.sim + ".npz")]
    elif c.file_list:
        with open(c.file_list) as f:
            ids = [ln.strip() for ln in f if ln.strip() and not ln.lstrip().startswith("#")]
        files = [os.path.join(cfg.data.dir, i if i.endswith(".npz") else i + ".npz") for i in ids]
    else:
        if c.split not in SPLIT_PRESETS:
            raise ValueError(f"infer.split={c.split!r} not in {sorted(SPLIT_PRESETS)}")
        if c.category and c.category not in splits["per_category"]:
            raise ValueError(f"infer.category={c.category!r} not in {sorted(splits['per_category'])}")
        files = []
        for s in SPLIT_PRESETS[c.split]:
            files += splits["per_category"][c.category][s] if c.category else splits[s]
    if c.max_samples is not None:
        files = files[:int(c.max_samples)]
    return files


def run_inference(cfg, model, splits, rank: int = 0, world_size: int = 1):
    """Roll out the selected trajectories (sharded across ranks) and export them."""
    files = select_files(cfg, splits)
    write = not bool(cfg.runtime.dry_run)
    if rank == 0:
        print(f"Rolling out {len(files)} trajectories" + (f" -> {cfg.infer.out_dir}" if write else " (dry run)"))
    for i in range(rank, len(files), world_size):
        start = time.time()
        pred = rollout(model, files[i], cfg, step_size=cfg.infer.infer_step_size)
        msg = f"[{i + 1}/{len(files)}] {os.path.relpath(files[i], cfg.data.dir)}"
        if write:
            ccd_on = cfg.ccd_postprocess.enabled and ccd_params_for(cfg, files[i]).enabled
            base = export_sequences(files[i], pred, cfg.infer.out_dir, "pred_ccd_post" if ccd_on else "pred")
            msg += f" -> {base}"
        print(msg + f" ({time.time() - start:.1f}s)", flush=True)
