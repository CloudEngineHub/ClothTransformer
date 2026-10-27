"""Dataset loading, train/val/test splits and batching.

Every simulation is one ``sim_XXXXX.npz`` file (see the dataset card on Hugging
Face for the full schema). The fields used here are:

    traj                (T, N_v, 3)  cloth vertex positions
    traj_vel            (T, N_v, 3)  per-frame cloth displacement
    initial             (N_v, 6)     rest positions [0:3] and UVs [3:6]
    edges / triangles                cloth topology
    collision_vertices  (T, N_c, 3)  collider vertex positions
    collision_vel       (T, N_c, 3)  per-frame collider displacement
    collision_triangles / collision_edges   collider topology
"""
import glob
import json
import math
import os
import random
from functools import lru_cache

import numpy as np
import torch
from torch.utils.data import Dataset


def load_sim(path: str, dt: float) -> dict:
    """Load one simulation. Velocities are returned in world units per second."""
    with np.load(path) as data:
        traj = data["traj"][:, :, :3]
        if "traj_vel" in data:
            traj_vel = data["traj_vel"][:, :, :3] / dt
        else:
            traj_vel = np.zeros_like(traj)
            traj_vel[1:] = (traj[1:] - traj[:-1]) / dt

        col_verts = data["collision_vertices"]
        col_tris = data["collision_triangles"]
        tri_idx = col_tris.flatten()
        # Per-frame triangle soup: (T, N_tris, 9) = three corners per triangle.
        collision = col_verts[:, tri_idx].reshape(col_verts.shape[0], -1, 9)
        if "collision_vel" in data:
            col_vel_verts = data["collision_vel"][:, :, :3] / dt
            collision_vel = col_vel_verts[:, tri_idx].reshape(col_verts.shape[0], -1, 9)
        else:
            collision_vel = np.zeros_like(collision)

        return {
            "traj": traj,
            "traj_vel": traj_vel,
            "collision": collision,
            "collision_vel": collision_vel,
            "rest_pos": data["initial"][:, :3],
            "edges": data["edges"],
            "triangles": data["triangles"] if "triangles" in data else np.zeros((0, 3), dtype=np.int32),
            "collision_vertices": col_verts,
            "collision_triangles": col_tris,
            "collision_edges": (data["collision_edges"] if "collision_edges" in data
                                else np.zeros((0, 2), dtype=np.int32)),
        }


def build_adjacency(edges: torch.Tensor, n_v: int) -> torch.Tensor:
    """Row-normalized (mean-aggregation) adjacency matrix of an undirected mesh."""
    adj = torch.zeros((n_v, n_v), dtype=torch.float32)
    valid = edges[(edges[:, 0] < n_v) & (edges[:, 1] < n_v)]
    if len(valid) > 0:
        adj[valid[:, 0], valid[:, 1]] = 1.0
        adj[valid[:, 1], valid[:, 0]] = 1.0
    degree = adj.sum(dim=1, keepdim=True)
    degree[degree == 0] = 1.0
    return adj / degree


# ----------------------------------------------------------------------------
# Splits
# ----------------------------------------------------------------------------

def discover_categories(root_dir: str) -> dict:
    """Map each sub-directory containing ``sim_*.npz`` files to its sorted file list.
    Falls back to ``root_dir`` itself as a single category."""
    categories = {}
    for sub in sorted(d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))):
        files = sorted(glob.glob(os.path.join(root_dir, sub, "sim_*.npz")))
        if files:
            categories[sub] = files
    if not categories:
        files = sorted(glob.glob(os.path.join(root_dir, "sim_*.npz")))
        if files:
            categories["default"] = files
    return categories


def _resolve(path: str) -> str:
    """Resolve repo-relative paths (e.g. the bundled split file)."""
    if os.path.isabs(path) or os.path.exists(path):
        return path
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(repo_root, path)


def build_splits(cfg, verbose: bool = True) -> dict:
    """Return {"train"|"val"|"test": [npz paths], "per_category": {cat: {split: [paths]}}}.

    Uses the fixed split ``data.split_file`` if set; otherwise splits every
    category at random with ``data.val_ratio`` / ``data.test_ratio``.
    """
    data_dir = cfg.data.dir
    split_file = str(cfg.data.split_file or "")
    to_path = lambda rel_id: os.path.join(data_dir, rel_id + ".npz")

    if split_file:
        split_file = _resolve(split_file)
        with open(split_file) as f:
            split_json = json.load(f)
        per_category = {cat: {k: [to_path(i) for i in ids] for k, ids in spl.items()}
                        for cat, spl in split_json["per_category"].items()}
        missing = [p for spl in per_category.values() for ps in spl.values() for p in ps
                   if not os.path.isfile(p)]
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} file(s) of {split_file} not found under data.dir={data_dir} "
                f"(e.g. {missing[:3]}). Download the dataset (scripts/download_data.py) or set "
                f"data.split_file=\"\" to split your own data at random.")
        source = split_file
    else:
        seed = int(cfg.training.seed)
        per_category = {}
        for cat, files in discover_categories(data_dir).items():
            n = len(files)
            perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
            n_val = min(max(1, int(n * float(cfg.data.val_ratio))), n)
            n_test = min(max(1, int(n * float(cfg.data.test_ratio))), n - n_val)
            per_category[cat] = {
                "train": [files[i] for i in perm[n_val + n_test:]],
                "val": [files[i] for i in perm[:n_val]],
                "test": [files[i] for i in perm[n_val:n_val + n_test]],
            }
        if not per_category:
            raise FileNotFoundError(f"No sim_*.npz files found under data.dir={data_dir}")
        source = f"random split of {data_dir}"

    splits = {k: [p for spl in per_category.values() for p in spl[k]] for k in ("train", "val", "test")}
    splits["per_category"] = per_category
    if verbose:
        print(f"Split: {source}")
        for cat, spl in per_category.items():
            print(f"  {cat}: train={len(spl['train'])}, val={len(spl['val'])}, test={len(spl['test'])}")
    return splits


def split_to_json(splits: dict, data_dir: str) -> dict:
    """Inverse of ``build_splits``: express a split as data-dir-relative ids."""
    rel = lambda p: os.path.relpath(p, data_dir)[:-len(".npz")]
    out = {k: [rel(p) for p in splits[k]] for k in ("train", "val", "test")}
    out["per_category"] = {cat: {k: [rel(p) for p in v] for k, v in spl.items()}
                           for cat, spl in splits["per_category"].items()}
    return out


# ----------------------------------------------------------------------------
# Training dataset
# ----------------------------------------------------------------------------

class ClothDataset(Dataset):
    """Sliding windows of ``seq_len`` consecutive frames over all trajectories."""

    def __init__(self, file_paths, seq_len: int, dt: float, velocity_scale: float,
                 augment: bool = False, translate_range: float = 0.5):
        super().__init__()
        self.file_paths = list(file_paths)
        self.seq_len = seq_len
        self.dt = dt
        self.velocity_scale = velocity_scale
        self.augment = augment
        self.translate_range = translate_range
        if self.file_paths:
            with np.load(self.file_paths[0]) as data:
                n_frames = data["traj"].shape[0]  # all trajectories share the same length
            self.windows_per_traj = max(n_frames - seq_len + 1, 0)
        else:
            self.windows_per_traj = 0

    @lru_cache(maxsize=128)
    def _load(self, path):
        return load_sim(path, self.dt)

    def __len__(self):
        return len(self.file_paths) * self.windows_per_traj

    def __getitem__(self, idx):
        sim = self._load(self.file_paths[idx // self.windows_per_traj])
        s = idx % self.windows_per_traj
        e = s + self.seq_len
        f32 = torch.float32

        traj = torch.from_numpy(sim["traj"][s:e]).to(f32)
        traj_vel = torch.from_numpy(sim["traj_vel"][s:e]).to(f32) * self.velocity_scale
        collision = torch.from_numpy(sim["collision"][s:e]).to(f32)
        collision_vel = torch.from_numpy(sim["collision_vel"][s:e]).to(f32) * self.velocity_scale
        rest_pos = torch.from_numpy(sim["rest_pos"]).to(f32)
        edges = torch.from_numpy(sim["edges"]).to(torch.long)
        cloth_tris = torch.from_numpy(sim["triangles"]).to(torch.long)

        if self.augment:
            # Random rotation about the (vertical) Y axis + random XZ translation.
            theta = random.random() * 2 * math.pi
            c, si = math.cos(theta), math.sin(theta)
            rot = torch.tensor([[c, 0, si], [0, 1, 0], [-si, 0, c]], dtype=f32)
            offset = torch.tensor([(random.random() * 2 - 1) * self.translate_range, 0.0,
                                   (random.random() * 2 - 1) * self.translate_range], dtype=f32)
            T, n_tris, _ = collision.shape
            traj = traj @ rot.T + offset
            traj_vel = traj_vel @ rot.T
            collision = (collision.reshape(T, n_tris, 3, 3) @ rot.T + offset).reshape(T, n_tris, 9)
            collision_vel = (collision_vel.reshape(T, n_tris, 3, 3) @ rot.T).reshape(T, n_tris, 9)
            rest_pos = rest_pos @ rot.T + offset

        return traj, collision, traj_vel, collision_vel, rest_pos, edges, cloth_tris


def cloth_collate_fn(batch):
    """Pad meshes of different sizes to the largest one in the batch.

    Returns a dict with padded tensors, padding masks (True = valid), the
    row-normalized adjacency matrices and the per-sample topology lists.
    """
    B = len(batch)
    T = batch[0][0].shape[0]
    max_v = max(item[0].shape[1] for item in batch)
    max_tris = max(item[1].shape[1] for item in batch)

    out = {
        "traj": torch.zeros(B, T, max_v, 3),
        "collision": torch.zeros(B, T, max_tris, 9),
        "traj_vel": torch.zeros(B, T, max_v, 3),
        "collision_vel": torch.zeros(B, T, max_tris, 9),
        "rest_pos": torch.zeros(B, max_v, 3),
        "cloth_mask": torch.zeros(B, max_v, dtype=torch.bool),
        "collision_mask": torch.zeros(B, max_tris, dtype=torch.bool),
        "adj": torch.zeros(B, max_v, max_v),
        "edges": [],
        "cloth_tris": [],
        "n_verts": [],
    }
    for i, (traj, collision, traj_vel, collision_vel, rest_pos, edges, cloth_tris) in enumerate(batch):
        n_v, n_tris = traj.shape[1], collision.shape[1]
        out["traj"][i, :, :n_v] = traj
        out["collision"][i, :, :n_tris] = collision
        out["traj_vel"][i, :, :n_v] = traj_vel
        out["collision_vel"][i, :, :n_tris] = collision_vel
        out["rest_pos"][i, :n_v] = rest_pos
        out["cloth_mask"][i, :n_v] = True
        out["collision_mask"][i, :n_tris] = True
        out["adj"][i, :n_v, :n_v] = build_adjacency(edges, n_v)
        out["edges"].append(edges[(edges[:, 0] < n_v) & (edges[:, 1] < n_v)])
        out["cloth_tris"].append(cloth_tris)
        out["n_verts"].append(n_v)
    return out
