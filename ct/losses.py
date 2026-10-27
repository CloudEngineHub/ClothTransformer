"""Training losses (the differentiable CCD loss lives in ``ct.ccd.loss``)."""
import torch
import torch.nn.functional as F


def smooth_max_loss(pred, tgt, beta: float = 10.0):
    """Numerically stable LogSumExp soft maximum of the element-wise squared error."""
    y = beta * (pred.float() - tgt.float()) ** 2
    with torch.no_grad():
        c = torch.max(y)
    return (c + torch.log(torch.mean(torch.exp(y - c)))) / beta


def reconstruction_loss(pred, tgt, mse_weight: float = 0.9, max_weight: float = 0.1, beta: float = 10.0):
    """MSE blended with a soft maximum of the squared error.

    Returns (loss, mse, max squared error).
    """
    pred, tgt = pred.float(), tgt.float()
    mse = F.mse_loss(pred, tgt)
    loss = mse_weight * mse + max_weight * smooth_max_loss(pred, tgt, beta)
    return loss, mse, torch.max((pred - tgt) ** 2)


def _face_centers_normals(v0, v1, v2, eps):
    centers = (v0 + v1 + v2) / 3.0
    normals = torch.cross(v1 - v0, v2 - v0, dim=-1)
    normals = normals / normals.norm(dim=-1, keepdim=True).clamp(min=eps)
    return centers, normals


def contact_loss(cloth_verts, collision_tris, thresh: float, knn: int = 1, max_dist: float = 0.05,
                 depth_cap: float = 0.01, eps: float = 1e-7):
    """Cubic penalty on cloth vertices penetrating the collider (discrete check).

    For every cloth vertex within ``max_dist`` of the collider, the signed distance
    to the nearest collider face (by face center) is measured along the face
    normal; the penetration ``clamp(thresh - d, 0, depth_cap)`` is penalized with
    its cube.

    Args:
        cloth_verts: (N_v, 3) predicted cloth vertices (differentiable)
        collision_tris: (N_tris, 9) collider triangle corners
    """
    zero = torch.tensor(0.0, device=cloth_verts.device, dtype=cloth_verts.dtype, requires_grad=True)
    n_tris = collision_tris.shape[0]
    if n_tris == 0:
        return zero
    tri = collision_tris.detach().reshape(n_tris, 3, 3)
    centers, normals = _face_centers_normals(tri[:, 0], tri[:, 1], tri[:, 2], eps)

    with torch.no_grad():
        dists = torch.cdist(cloth_verts.detach().float(), centers.float())
        min_dists, nn_idx = dists.topk(min(knn, n_tris), dim=-1, largest=False)
    close = (min_dists < max_dist).squeeze(-1)
    if not close.any():
        return zero

    offset = cloth_verts[close].unsqueeze(1) - centers[nn_idx[close]]
    signed_dist = (offset * normals[nn_idx[close]]).sum(dim=-1)
    return torch.clamp(thresh - signed_dist, min=0.0, max=depth_cap).pow(3).mean()


def self_contact_loss(cloth_verts, cloth_faces, thresh: float, knn: int = 5, eps: float = 1e-7):
    """Cubic penalty on cloth vertices closer than ``thresh`` to non-incident cloth faces.

    Args:
        cloth_verts: (N_v, 3) predicted cloth vertices (differentiable)
        cloth_faces: (N_f, 3) cloth triangle indices
    """
    n_v, n_f = cloth_verts.shape[0], cloth_faces.shape[0]
    if n_f == 0 or n_v == 0:
        return torch.tensor(0.0, device=cloth_verts.device, dtype=cloth_verts.dtype, requires_grad=True)

    # Face geometry is detached; gradients flow through the query vertices only.
    v = cloth_verts.detach()
    centers, normals = _face_centers_normals(v[cloth_faces[:, 0]], v[cloth_faces[:, 1]],
                                             v[cloth_faces[:, 2]], eps)

    k_search = min(knn + 3, n_f)  # search a few extra faces to compensate for filtering
    with torch.no_grad():
        dists = torch.cdist(v.float(), centers.float())
        _, nn_idx = dists.topk(k_search, dim=-1, largest=False)
        # Exclude faces incident to the query vertex.
        vert_ids = torch.arange(n_v, device=cloth_verts.device).view(-1, 1, 1)
        own_face = (cloth_faces[nn_idx] == vert_ids).any(dim=-1)
        nn_dists = dists.gather(1, nn_idx)
        nn_dists[own_face] = float('inf')
        _, rerank = nn_dists.topk(min(knn, k_search), dim=-1, largest=False)
        final_idx = nn_idx.gather(1, rerank)

    offset = cloth_verts.unsqueeze(1) - centers[final_idx]
    signed_dist = (offset * normals[final_idx]).sum(dim=-1)
    return torch.clamp(thresh - signed_dist.abs(), min=0.0).pow(3).sum()


def linear_ramp(step: int, start_step: int, ramp_steps: int, start_value: float, end_value: float):
    progress = min(max(step - start_step, 0) / max(ramp_steps, 1), 1.0)
    return start_value + (end_value - start_value) * progress
