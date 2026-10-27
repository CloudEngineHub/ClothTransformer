"""Differentiable CCD (Continuous Collision Detection) loss.

Mirrors the five collision types of ``ct.ccd.postprocess`` but returns a
differentiable penalty instead of modifying vertex positions:

  1. VF      cloth vertex     vs collider triangle
  2. EE      cloth edge       vs collider edge
  3. TV      collider vertex  vs cloth triangle
  4. Self-VF cloth vertex     vs cloth triangle
  5. Self-EE cloth edge       vs cloth edge

Broad-phase (AABB) and CCD detection run without gradients. For every detected
collision a differentiable penalty is computed on the end-state positions
(``cloth_verts_end``), which are part of the computation graph. During training
only the self-collision types are enabled, since cloth-collider contact is
already supervised by the contact loss.
"""
import torch
import torch.nn as nn

from ct.ccd.primitives import (
    edge_edge_ccd,
    find_edge_edge_pairs,
    find_vertex_triangle_pairs,
    point_triangle_ccd,
)


# ======================== Differentiable helpers ======================== #

def _compute_triangle_normal(t0: torch.Tensor, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Compute unit triangle normals.  Input/output shape: (N, 3)."""
    e0 = t1 - t0
    e1 = t2 - t0
    normal = torch.linalg.cross(e0, e1)
    norm = normal.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return normal / norm


def _point_triangle_signed_distance(p: torch.Tensor, t0: torch.Tensor,
                                     t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
    """Signed distance from points to triangle planes.  All (N, 3) -> (N,)."""
    normal = _compute_triangle_normal(t0, t1, t2)
    return (p - t0).mul(normal).sum(dim=-1)


def _edge_edge_signed_distance(p0: torch.Tensor, p1: torch.Tensor,
                                q0: torch.Tensor, q1: torch.Tensor):
    """
    Signed distance between two edge pairs using cross-product normal.
    All inputs (N, 3).  Returns (signed_dist (N,), valid_mask (N,)).
    """
    edge_p = p1 - p0
    edge_q = q1 - q0
    cross = torch.linalg.cross(edge_p, edge_q)
    cross_norm = cross.norm(dim=-1, keepdim=True)
    valid = (cross_norm.squeeze(-1) > 1e-8)
    cross_unit = cross / cross_norm.clamp(min=1e-8)
    connect = p0 - q0
    signed_dist = (connect * cross_unit).sum(dim=-1)
    return signed_dist, valid


def _edge_edge_min_distance(p0: torch.Tensor, p1: torch.Tensor,
                             q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    """
    Minimum distance between two line segments (differentiable).
    All inputs (N, 3) -> (N,).
    """
    d1 = p1 - p0
    d2 = q1 - q0
    r = p0 - q0

    a = (d1 * d1).sum(dim=-1)  # (N,)
    b = (d1 * d2).sum(dim=-1)
    c = (d2 * d2).sum(dim=-1)
    d = (d1 * r).sum(dim=-1)
    e = (d2 * r).sum(dim=-1)

    denom = a * c - b * b
    # For near-parallel edges, fall back to s=0
    parallel = (denom < 1e-12)
    denom_safe = denom.clamp(min=1e-12)

    s = ((b * e - c * d) / denom_safe).clamp(0.0, 1.0)
    t = ((a * e - b * d) / denom_safe).clamp(0.0, 1.0)

    s = torch.where(parallel, torch.zeros_like(s), s)
    t = torch.where(parallel, (e / c.clamp(min=1e-12)).clamp(0.0, 1.0), t)

    closest1 = p0 + s.unsqueeze(-1) * d1
    closest2 = q0 + t.unsqueeze(-1) * d2
    return (closest1 - closest2).norm(dim=-1)


# ======================== Main CCD Loss Class ======================== #

class CCDLoss(nn.Module):
    """
    Differentiable collision loss based on CCD detection.

    Usage:
        loss_module = CCDLoss()
        ccd_loss, info = loss_module(
            cloth_verts_start, cloth_verts_end,
            col_verts_start, col_verts_end,
            cloth_edges, col_edges, col_tris, cloth_tris,
            n_cloth_v, n_col_v, ...
        )
        total_loss = pred_loss + weight * ccd_loss
    """

    def __init__(self,
                 margin: float = 0.01,
                 safe_margin: float = 0.001,
                 max_pairs: int = 100000,
                 loss_factor: float = 1.0,
                 enable_vf: bool = False,
                 enable_ee: bool = False,
                 enable_tv: bool = False,
                 enable_self_vf: bool = True,
                 enable_self_ee: bool = True):
        super().__init__()
        self.margin = margin
        self.safe_margin = safe_margin
        self.max_pairs = max_pairs
        self.loss_factor = loss_factor
        self.enable_vf = enable_vf
        self.enable_ee = enable_ee
        self.enable_tv = enable_tv
        self.enable_self_vf = enable_self_vf
        self.enable_self_ee = enable_self_ee

    def forward(self,
                cloth_verts_start: torch.Tensor,
                cloth_verts_end: torch.Tensor,
                col_verts_start: torch.Tensor,
                col_verts_end: torch.Tensor,
                cloth_edges: torch.Tensor,
                col_edges: torch.Tensor,
                col_tris: torch.Tensor,
                cloth_tris: torch.Tensor,
                n_cloth_v: int,
                n_col_v: int,
                n_cloth_e: int,
                n_col_e: int,
                n_col_f: int,
                n_cloth_f: int) -> tuple:
        """
        Compute differentiable CCD collision loss.

        The inputs cloth_verts_end should be the predicted positions that
        are part of the computation graph.  cloth_verts_start and col_*
        are treated as fixed (detached or ground truth).

        Returns:
            loss: scalar tensor (differentiable w.r.t. cloth_verts_end)
            info: dict with per-type collision counts
        """
        device = cloth_verts_end.device
        zero_loss = torch.tensor(0.0, device=device, dtype=cloth_verts_end.dtype)
        info = {'vf': 0, 'ee': 0, 'tv': 0, 'self_vf': 0, 'self_ee': 0}

        if n_cloth_v == 0:
            return zero_loss, info

        # Trim to actual sizes
        cloth_v_s = cloth_verts_start[:n_cloth_v].detach()
        cloth_v_e = cloth_verts_end[:n_cloth_v]   # keep grad!
        col_v_s = col_verts_start[:n_col_v].detach()
        col_v_e = col_verts_end[:n_col_v].detach()
        cloth_e = cloth_edges[:n_cloth_e]
        col_e = col_edges[:n_col_e]
        col_f = col_tris[:n_col_f]
        cloth_f = cloth_tris[:n_cloth_f]

        total_loss = zero_loss

        # ============ 1. VF: Cloth Vertex vs Collision Triangle ============
        if self.enable_vf and n_col_f > 0:
            vf_loss, n_vf = self._vf_loss(cloth_v_s, cloth_v_e, col_v_s, col_v_e, col_f)
            total_loss = total_loss + vf_loss
            info['vf'] = n_vf

        # ============ 2. EE: Cloth Edge vs Collision Edge ============
        if self.enable_ee and n_cloth_e > 0 and n_col_e > 0:
            ee_loss, n_ee = self._ee_loss(
                cloth_v_s, cloth_v_e, cloth_e,
                col_v_s, col_v_e, col_e,
                same_mesh=False)
            total_loss = total_loss + ee_loss
            info['ee'] = n_ee

        # ============ 3. TV: Collision Vertex vs Cloth Triangle ============
        if self.enable_tv and n_cloth_f > 0 and n_col_v > 0:
            tv_loss, n_tv = self._tv_loss(cloth_v_s, cloth_v_e, cloth_f, col_v_s, col_v_e)
            total_loss = total_loss + tv_loss
            info['tv'] = n_tv

        # ============ 4. Self-VF: Cloth Vertex vs Cloth Triangle ============
        if self.enable_self_vf and n_cloth_f > 0 and n_cloth_v > 0:
            self_vf_loss, n_svf = self._self_vf_loss(cloth_v_s, cloth_v_e, cloth_f)
            total_loss = total_loss + self_vf_loss
            info['self_vf'] = n_svf

        # ============ 5. Self-EE: Cloth Edge vs Cloth Edge ============
        if self.enable_self_ee and n_cloth_e > 0:
            self_ee_loss, n_see = self._ee_loss(
                cloth_v_s, cloth_v_e, cloth_e,
                cloth_v_s, cloth_v_e, cloth_e,
                same_mesh=True)
            total_loss = total_loss + self_ee_loss
            info['self_ee'] = n_see

        return total_loss * self.loss_factor, info

    # -------------------- Per-type loss computation -------------------- #

    def _vf_loss(self, cloth_v_s, cloth_v_e, col_v_s, col_v_e, col_f):
        """Cloth vertex vs collision triangle — differentiable penalty."""
        device = cloth_v_e.device

        col_t0_s = col_v_s[col_f[:, 0]]
        col_t0_e = col_v_e[col_f[:, 0]]
        col_t1_s = col_v_s[col_f[:, 1]]
        col_t1_e = col_v_e[col_f[:, 1]]
        col_t2_s = col_v_s[col_f[:, 2]]
        col_t2_e = col_v_e[col_f[:, 2]]

        # Broad-phase
        with torch.no_grad():
            vt_pairs = find_vertex_triangle_pairs(
                cloth_v_s, cloth_v_e,
                col_t0_s, col_t0_e, col_t1_s, col_t1_e, col_t2_s, col_t2_e,
                self.margin, self.max_pairs)

        if vt_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        v_idx = vt_pairs[:, 0]
        t_idx = vt_pairs[:, 1]

        # CCD detection (non-differentiable)
        with torch.no_grad():
            t_impact, is_colliding = point_triangle_ccd(
                cloth_v_s[v_idx], cloth_v_e[v_idx].detach(),
                col_t0_s[t_idx], col_t0_e[t_idx],
                col_t1_s[t_idx], col_t1_e[t_idx],
                col_t2_s[t_idx], col_t2_e[t_idx])

        if not is_colliding.any():
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        # Filter to colliding pairs
        col_v_idx = v_idx[is_colliding]
        col_t_idx = t_idx[is_colliding]

        # Compute signed distance at t=1 — DIFFERENTIABLE w.r.t. cloth_v_e
        with torch.no_grad():
            dist_start = _point_triangle_signed_distance(
                cloth_v_s[col_v_idx],
                col_t0_s[col_t_idx], col_t1_s[col_t_idx], col_t2_s[col_t_idx])
            target_side = torch.where(dist_start >= 0,
                                      torch.ones_like(dist_start),
                                      -torch.ones_like(dist_start))

        # Differentiable signed distance at end position
        dist_end = _point_triangle_signed_distance(
            cloth_v_e[col_v_idx],
            col_t0_e[col_t_idx], col_t1_e[col_t_idx], col_t2_e[col_t_idx])

        # Penetration: vertex should be on target_side with margin
        # signed_correct = dist_end * target_side  (positive = safe, negative = penetrating)
        signed_correct = dist_end * target_side

        # Penalty: penalize when signed_correct < safe_margin
        # penalty = ReLU(safe_margin - signed_correct)^2
        violation = torch.relu(self.safe_margin - signed_correct)
        loss = (violation ** 2).mean()

        return loss, is_colliding.sum().item()

    def _ee_loss(self, v1_s, v1_e, edges1, v2_s, v2_e, edges2, same_mesh=False):
        """Edge-edge collision loss (cloth vs collision, or self-collision)."""
        device = v1_e.device

        e1_p0_s = v1_s[edges1[:, 0]]
        e1_p0_e = v1_e[edges1[:, 0]]
        e1_p1_s = v1_s[edges1[:, 1]]
        e1_p1_e = v1_e[edges1[:, 1]]

        e2_p0_s = v2_s[edges2[:, 0]]
        e2_p0_e = v2_e[edges2[:, 0]]
        e2_p1_s = v2_s[edges2[:, 1]]
        e2_p1_e = v2_e[edges2[:, 1]]

        # Broad-phase
        with torch.no_grad():
            ee_pairs = find_edge_edge_pairs(
                e1_p0_s, e1_p0_e.detach(), e1_p1_s, e1_p1_e.detach(),
                e2_p0_s, e2_p0_e.detach(), e2_p1_s, e2_p1_e.detach(),
                self.margin, self.max_pairs, same_mesh=same_mesh)

        if ee_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=device, dtype=v1_e.dtype), 0

        ei1 = ee_pairs[:, 0]
        ei2 = ee_pairs[:, 1]

        # Filter out shared-vertex pairs for same_mesh
        if same_mesh:
            e1_v0 = edges1[ei1, 0]
            e1_v1 = edges1[ei1, 1]
            e2_v0 = edges2[ei2, 0]
            e2_v1 = edges2[ei2, 1]
            valid = (e1_v0 != e2_v0) & (e1_v0 != e2_v1) & \
                    (e1_v1 != e2_v0) & (e1_v1 != e2_v1)
            if not valid.any():
                return torch.tensor(0.0, device=device, dtype=v1_e.dtype), 0
            ei1 = ei1[valid]
            ei2 = ei2[valid]

        # CCD detection (non-differentiable)
        with torch.no_grad():
            t_impact, is_colliding = edge_edge_ccd(
                e1_p0_s[ei1], e1_p0_e[ei1].detach(),
                e1_p1_s[ei1], e1_p1_e[ei1].detach(),
                e2_p0_s[ei2], e2_p0_e[ei2].detach(),
                e2_p1_s[ei2], e2_p1_e[ei2].detach())

        if not is_colliding.any():
            return torch.tensor(0.0, device=device, dtype=v1_e.dtype), 0

        col_ei1 = ei1[is_colliding]
        col_ei2 = ei2[is_colliding]

        # Compute signed distance at start to determine target side
        with torch.no_grad():
            dist_start, valid_start = _edge_edge_signed_distance(
                e1_p0_s[col_ei1], e1_p1_s[col_ei1],
                e2_p0_s[col_ei2], e2_p1_s[col_ei2])
            target_side = torch.where(dist_start >= 0,
                                      torch.ones_like(dist_start),
                                      -torch.ones_like(dist_start))

        # Differentiable: minimum distance at end position
        min_dist = _edge_edge_min_distance(
            v1_e[edges1[col_ei1, 0]], v1_e[edges1[col_ei1, 1]],
            v2_e[edges2[col_ei2, 0]], v2_e[edges2[col_ei2, 1]])

        # Differentiable: signed distance at end position
        signed_end, valid_end = _edge_edge_signed_distance(
            v1_e[edges1[col_ei1, 0]], v1_e[edges1[col_ei1, 1]],
            v2_e[edges2[col_ei2, 0]], v2_e[edges2[col_ei2, 1]])

        # Align sign direction with start
        with torch.no_grad():
            # Check if the cross-product direction flipped
            cross_s = torch.linalg.cross(
                e1_p1_s[col_ei1] - e1_p0_s[col_ei1],
                e2_p1_s[col_ei2] - e2_p0_s[col_ei2])
            cross_e = torch.linalg.cross(
                v1_e[edges1[col_ei1, 1]] - v1_e[edges1[col_ei1, 0]],
                v2_e[edges2[col_ei2, 1]] - v2_e[edges2[col_ei2, 0]])
            alignment = (cross_s * cross_e).sum(dim=-1)
            flip_mask = alignment < 0

        signed_corrected = torch.where(flip_mask, -signed_end, signed_end)
        signed_correct = signed_corrected * target_side

        # Penalty: penalize when on wrong side or too close
        violation = torch.relu(self.safe_margin - signed_correct)
        # Also penalize based on actual minimum distance being too small
        dist_violation = torch.relu(self.safe_margin - min_dist)
        # Use the maximum of both penalties
        combined_violation = torch.max(violation, dist_violation)
        loss = (combined_violation ** 2).mean()

        return loss, is_colliding.sum().item()

    def _tv_loss(self, cloth_v_s, cloth_v_e, cloth_f, col_v_s, col_v_e):
        """Collision vertex vs cloth triangle — differentiable penalty.
        The cloth triangle vertices are differentiable."""
        device = cloth_v_e.device

        cloth_t0_s = cloth_v_s[cloth_f[:, 0]]
        cloth_t0_e = cloth_v_e[cloth_f[:, 0]]
        cloth_t1_s = cloth_v_s[cloth_f[:, 1]]
        cloth_t1_e = cloth_v_e[cloth_f[:, 1]]
        cloth_t2_s = cloth_v_s[cloth_f[:, 2]]
        cloth_t2_e = cloth_v_e[cloth_f[:, 2]]

        # Broad-phase
        with torch.no_grad():
            tv_pairs = find_vertex_triangle_pairs(
                col_v_s, col_v_e,
                cloth_t0_s, cloth_t0_e.detach(),
                cloth_t1_s, cloth_t1_e.detach(),
                cloth_t2_s, cloth_t2_e.detach(),
                self.margin, self.max_pairs)

        if tv_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        v_idx = tv_pairs[:, 0]  # collision vertex indices
        t_idx = tv_pairs[:, 1]  # cloth triangle indices

        # CCD detection (non-differentiable)
        with torch.no_grad():
            t_impact, is_colliding = point_triangle_ccd(
                col_v_s[v_idx], col_v_e[v_idx],
                cloth_t0_s[t_idx], cloth_t0_e[t_idx].detach(),
                cloth_t1_s[t_idx], cloth_t1_e[t_idx].detach(),
                cloth_t2_s[t_idx], cloth_t2_e[t_idx].detach())

        if not is_colliding.any():
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        col_v_idx = v_idx[is_colliding]
        col_t_idx = t_idx[is_colliding]

        with torch.no_grad():
            dist_start = _point_triangle_signed_distance(
                col_v_s[col_v_idx],
                cloth_t0_s[col_t_idx], cloth_t1_s[col_t_idx], cloth_t2_s[col_t_idx])
            target_side = torch.where(dist_start >= 0,
                                      torch.ones_like(dist_start),
                                      -torch.ones_like(dist_start))

        # Differentiable: signed distance at end position
        # Here cloth triangle vertices are differentiable
        dist_end = _point_triangle_signed_distance(
            col_v_e[col_v_idx],
            cloth_t0_e[col_t_idx], cloth_t1_e[col_t_idx], cloth_t2_e[col_t_idx])

        signed_correct = dist_end * target_side
        violation = torch.relu(self.safe_margin - signed_correct)
        loss = (violation ** 2).mean()

        return loss, is_colliding.sum().item()

    def _self_vf_loss(self, cloth_v_s, cloth_v_e, cloth_f):
        """Cloth vertex vs cloth triangle self-collision loss."""
        device = cloth_v_e.device

        cloth_t0_s = cloth_v_s[cloth_f[:, 0]]
        cloth_t0_e = cloth_v_e[cloth_f[:, 0]]
        cloth_t1_s = cloth_v_s[cloth_f[:, 1]]
        cloth_t1_e = cloth_v_e[cloth_f[:, 1]]
        cloth_t2_s = cloth_v_s[cloth_f[:, 2]]
        cloth_t2_e = cloth_v_e[cloth_f[:, 2]]

        # Broad-phase
        with torch.no_grad():
            vf_pairs = find_vertex_triangle_pairs(
                cloth_v_s, cloth_v_e,
                cloth_t0_s, cloth_t0_e.detach(),
                cloth_t1_s, cloth_t1_e.detach(),
                cloth_t2_s, cloth_t2_e.detach(),
                self.margin, self.max_pairs)

        if vf_pairs.shape[0] == 0:
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        v_idx = vf_pairs[:, 0]
        t_idx = vf_pairs[:, 1]

        # Filter out vertex-on-own-triangle pairs
        tri_v0 = cloth_f[t_idx, 0]
        tri_v1 = cloth_f[t_idx, 1]
        tri_v2 = cloth_f[t_idx, 2]
        valid = (v_idx != tri_v0) & (v_idx != tri_v1) & (v_idx != tri_v2)

        if not valid.any():
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        v_idx = v_idx[valid]
        t_idx = t_idx[valid]

        # CCD detection
        with torch.no_grad():
            t_impact, is_colliding = point_triangle_ccd(
                cloth_v_s[v_idx], cloth_v_e[v_idx].detach(),
                cloth_t0_s[t_idx], cloth_t0_e[t_idx].detach(),
                cloth_t1_s[t_idx], cloth_t1_e[t_idx].detach(),
                cloth_t2_s[t_idx], cloth_t2_e[t_idx].detach())

        if not is_colliding.any():
            return torch.tensor(0.0, device=device, dtype=cloth_v_e.dtype), 0

        col_v_idx = v_idx[is_colliding]
        col_t_idx = t_idx[is_colliding]

        with torch.no_grad():
            dist_start = _point_triangle_signed_distance(
                cloth_v_s[col_v_idx],
                cloth_t0_s[col_t_idx], cloth_t1_s[col_t_idx], cloth_t2_s[col_t_idx])
            target_side = torch.where(dist_start >= 0,
                                      torch.ones_like(dist_start),
                                      -torch.ones_like(dist_start))

        # Both vertex and triangle vertices are differentiable
        dist_end = _point_triangle_signed_distance(
            cloth_v_e[col_v_idx],
            cloth_t0_e[col_t_idx], cloth_t1_e[col_t_idx], cloth_t2_e[col_t_idx])

        signed_correct = dist_end * target_side
        violation = torch.relu(self.safe_margin - signed_correct)
        loss = (violation ** 2).mean()

        return loss, is_colliding.sum().item()
