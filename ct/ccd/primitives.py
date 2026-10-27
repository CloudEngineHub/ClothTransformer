"""Batched CCD primitives: broad-phase AABB pair search and narrow-phase
point-triangle / edge-edge continuous collision detection.

All narrow-phase tests assume linear motion over the time step t in [0, 1]. The
coplanarity condition is a cubic in t; its roots are found by splitting [0, 1]
into monotonic intervals (critical + inflection points) and running Newton
iterations inside each interval.
"""
from typing import Tuple

import torch


# ----------------------------------------------------------------------------
# Broad phase
# ----------------------------------------------------------------------------

_warned = set()


def _warn_once(kind: str, n: int, max_pairs: int):
    if kind not in _warned:
        _warned.add(kind)
        print(f"[CCD] {n} {kind} candidate pairs exceed max_pairs={max_pairs}; keeping the closest "
              f"ones (further warnings suppressed).")


def _aabb_overlap(min1, max1, min2, max2) -> torch.Tensor:
    """(N, 3) x (M, 3) boxes -> (N, M) boolean overlap matrix."""
    overlap = (min1.unsqueeze(1) <= max2.unsqueeze(0)) & (max1.unsqueeze(1) >= min2.unsqueeze(0))
    return overlap.all(dim=-1)


def find_vertex_triangle_pairs(v_start, v_end,
                               t0_start, t0_end, t1_start, t1_end, t2_start, t2_end,
                               margin: float = 0.05, max_pairs: int = 100000) -> torch.Tensor:
    """Candidate (vertex, triangle) pairs whose swept AABBs overlap. Returns (P, 2)."""
    if v_start.shape[0] == 0 or t0_start.shape[0] == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=v_start.device)

    v_min = torch.minimum(v_start, v_end) - margin
    v_max = torch.maximum(v_start, v_end) + margin
    tri = torch.stack([t0_start, t0_end, t1_start, t1_end, t2_start, t2_end], dim=1)
    t_min = tri.min(dim=1).values - margin
    t_max = tri.max(dim=1).values + margin
    pairs = torch.nonzero(_aabb_overlap(v_min, v_max, t_min, t_max), as_tuple=False)

    if pairs.shape[0] > max_pairs:
        _warn_once("vertex-triangle", pairs.shape[0], max_pairs)
        v_idx, t_idx = pairs[:, 0], pairs[:, 1]
        v_center = (v_start[v_idx] + v_end[v_idx]) / 2
        t_center = (t0_start[t_idx] + t0_end[t_idx] + t1_start[t_idx] + t1_end[t_idx]
                    + t2_start[t_idx] + t2_end[t_idx]) / 6
        dist_sq = ((v_center - t_center) ** 2).sum(dim=1)
        pairs = pairs[torch.sort(dist_sq).indices[:max_pairs]]
    return pairs


def find_edge_edge_pairs(e1_p0_start, e1_p0_end, e1_p1_start, e1_p1_end,
                         e2_p0_start, e2_p0_end, e2_p1_start, e2_p1_end,
                         margin: float = 0.05, max_pairs: int = 100000,
                         same_mesh: bool = False) -> torch.Tensor:
    """Candidate (edge, edge) pairs whose swept AABBs overlap. Returns (P, 2).

    With ``same_mesh=True`` only pairs with i < j are returned.
    """
    n1, n2 = e1_p0_start.shape[0], e2_p0_start.shape[0]
    if n1 == 0 or n2 == 0:
        return torch.zeros((0, 2), dtype=torch.long, device=e1_p0_start.device)

    e1 = torch.stack([e1_p0_start, e1_p0_end, e1_p1_start, e1_p1_end], dim=1)
    e2 = torch.stack([e2_p0_start, e2_p0_end, e2_p1_start, e2_p1_end], dim=1)
    overlap = _aabb_overlap(e1.min(dim=1).values - margin, e1.max(dim=1).values + margin,
                            e2.min(dim=1).values - margin, e2.max(dim=1).values + margin)
    if same_mesh:
        idx1 = torch.arange(n1, device=overlap.device).unsqueeze(1)
        idx2 = torch.arange(n2, device=overlap.device).unsqueeze(0)
        overlap = overlap & (idx1 < idx2)
    pairs = torch.nonzero(overlap, as_tuple=False)

    if pairs.shape[0] > max_pairs:
        _warn_once("edge-edge", pairs.shape[0], max_pairs)
        i1, i2 = pairs[:, 0], pairs[:, 1]
        c1 = (e1_p0_start[i1] + e1_p0_end[i1] + e1_p1_start[i1] + e1_p1_end[i1]) / 4
        c2 = (e2_p0_start[i2] + e2_p0_end[i2] + e2_p1_start[i2] + e2_p1_end[i2]) / 4
        dist_sq = ((c1 - c2) ** 2).sum(dim=1)
        pairs = pairs[torch.sort(dist_sq).indices[:max_pairs]]
    return pairs


# ----------------------------------------------------------------------------
# Narrow phase
# ----------------------------------------------------------------------------

def _solve_quadratic_stable(a, b, c):
    """Numerically stable roots of a x^2 + b x + c = 0 (batched)."""
    delta = b ** 2 - 4 * a * c
    has_real = delta >= 0
    sqrt_delta = torch.sqrt(torch.relu(delta))
    sgn_b = torch.sign(b)
    sgn_b[sgn_b == 0] = 1.0
    q = b + sgn_b * sqrt_delta
    x1 = -2 * c / (q + 1e-12)
    x2 = -q / (2 * a + 1e-12)
    return x1, x2, has_real


def _eval_cubic(coeffs, t):
    return coeffs[..., 0] * t ** 3 + coeffs[..., 1] * t ** 2 + coeffs[..., 2] * t + coeffs[..., 3]


def _eval_cubic_derivative(coeffs, t):
    return 3 * coeffs[..., 0] * t ** 2 + 2 * coeffs[..., 1] * t + coeffs[..., 2]


def _triple_prod(a, b, c):
    return torch.sum(a * torch.cross(b, c, dim=1), dim=1)


def _cubic_coeffs(m0_s, m0_v, m1_s, m1_v, m2_s, m2_v):
    """Coefficients of (m0_s + t m0_v) . ((m1_s + t m1_v) x (m2_s + t m2_v))."""
    a = _triple_prod(m0_v, m1_v, m2_v)
    b = (_triple_prod(m0_s, m1_v, m2_v) + _triple_prod(m0_v, m1_s, m2_v)
         + _triple_prod(m0_v, m1_v, m2_s))
    c = (_triple_prod(m0_v, m1_s, m2_s) + _triple_prod(m0_s, m1_v, m2_s)
         + _triple_prod(m0_s, m1_s, m2_v))
    d = _triple_prod(m0_s, m1_s, m2_s)
    return torch.stack([a, b, c, d], dim=1)


def _earliest_root_in_unit_interval(coeffs, max_iter: int):
    """Smallest root of the cubic in [0, 1]. Returns (t, found) with t=2 if none."""
    a, b, c = coeffs[:, 0], coeffs[:, 1], coeffs[:, 2]
    crit1, crit2, has_crit = _solve_quadratic_stable(3 * a, 2 * b, c)
    inflection = -b / (3 * a + 1e-12)
    zeros, ones = torch.zeros_like(a), torch.ones_like(a)
    crit1 = torch.where(has_crit, crit1, zeros)
    crit2 = torch.where(has_crit, crit2, zeros)

    # Split points of [0, 1] into monotonic sub-intervals.
    candidates = torch.stack([zeros, ones, crit1, crit2, inflection], dim=1)
    candidates, _ = torch.sort(torch.clamp(candidates, 0.0, 1.0), dim=1)
    vals = _eval_cubic(coeffs.unsqueeze(1), candidates)

    min_t = torch.ones_like(a) * 2.0
    found = torch.zeros_like(a, dtype=torch.bool)
    for i in range(4):
        t_lo, t_hi = candidates[:, i], candidates[:, i + 1]
        has_root = (vals[:, i] * vals[:, i + 1] <= 0) & ((t_hi - t_lo) > 1e-6)
        t_curr = (t_lo + t_hi) * 0.5
        for _ in range(max_iter):
            step = _eval_cubic(coeffs, t_curr) / (_eval_cubic_derivative(coeffs, t_curr) + 1e-12)
            t_next = torch.clamp(t_curr - step, t_lo, t_hi)
            t_curr = torch.where(has_root, t_next, t_curr)
        is_root = (torch.abs(_eval_cubic(coeffs, t_curr)) < 1e-4) & has_root
        update = is_root & (t_curr < min_t)
        min_t = torch.where(update, t_curr, min_t)
        found = found | update
    return min_t, found


def point_triangle_ccd(p_start, p_end, t0_start, t0_end, t1_start, t1_end, t2_start, t2_end,
                       eps: float = 1e-6, max_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Point-triangle CCD. All inputs (N, 3).

    Returns:
        t_impact: (N,) earliest time of impact in [0, 1], -1 if no collision.
        is_colliding: (N,) bool.
    """
    v_p = p_end - p_start
    v_t0 = t0_end - t0_start
    v_t1 = t1_end - t1_start
    v_t2 = t2_end - t2_start

    # Work in the frame of triangle vertex t0.
    coeffs = _cubic_coeffs(p_start - t0_start, v_p - v_t0,
                           t1_start - t0_start, v_t1 - v_t0,
                           t2_start - t0_start, v_t2 - v_t0)
    t_final, found = _earliest_root_in_unit_interval(coeffs, max_iter)

    # Barycentric inside-triangle test at the time of impact.
    tt = t_final.unsqueeze(1)
    p_f = p_start + tt * v_p
    a_f = t0_start + tt * v_t0
    b_f = t1_start + tt * v_t1
    c_f = t2_start + tt * v_t2
    v0, v1, v2 = b_f - a_f, c_f - a_f, p_f - a_f
    d00 = torch.sum(v0 * v0, dim=1)
    d01 = torch.sum(v0 * v1, dim=1)
    d11 = torch.sum(v1 * v1, dim=1)
    d20 = torch.sum(v2 * v0, dim=1)
    d21 = torch.sum(v2 * v1, dim=1)
    denom = d00 * d11 - d01 * d01
    valid_tri = denom > 1e-12
    v = (d11 * d20 - d01 * d21) / (denom + 1e-12)
    w = (d00 * d21 - d01 * d20) / (denom + 1e-12)
    u = 1.0 - v - w
    inside = (v >= -eps) & (w >= -eps) & (u >= -eps)

    is_colliding = found & valid_tri & inside & (t_final >= 0) & (t_final <= 1)
    t_final = torch.where(is_colliding, t_final, torch.tensor(-1.0, device=p_start.device))
    return t_final, is_colliding


def edge_edge_ccd(p0_start, p0_end, p1_start, p1_end,
                  q0_start, q0_end, q1_start, q1_end,
                  eps: float = 1e-6, max_iter: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    """Edge-edge CCD between edges (p0, p1) and (q0, q1). All inputs (N, 3).

    Returns:
        t_impact: (N,) earliest time of impact in [0, 1], -1 if no collision.
        is_colliding: (N,) bool.
    """
    v_p0 = p0_end - p0_start
    v_p1 = p1_end - p1_start
    v_q0 = q0_end - q0_start
    v_q1 = q1_end - q1_start

    # Coplanarity: (P1 - P0) . ((Q1 - Q0) x (Q0 - P0)) = 0.
    coeffs = _cubic_coeffs(p1_start - p0_start, v_p1 - v_p0,
                           q1_start - q0_start, v_q1 - v_q0,
                           q0_start - p0_start, v_q0 - v_p0)
    t_final, found = _earliest_root_in_unit_interval(coeffs, max_iter)

    # Closest-point parameters (u on edge p, v on edge q) at the time of impact,
    # solving u * a - v * b = c.
    tt = t_final.unsqueeze(1)
    p0_f = p0_start + tt * v_p0
    p1_f = p1_start + tt * v_p1
    q0_f = q0_start + tt * v_q0
    q1_f = q1_start + tt * v_q1
    vec_a, vec_b, vec_c = p1_f - p0_f, q1_f - q0_f, q0_f - p0_f
    d00 = torch.sum(vec_a * vec_a, dim=1)
    d01 = torch.sum(vec_a * vec_b, dim=1)
    d11 = torch.sum(vec_b * vec_b, dim=1)
    dc0 = torch.sum(vec_c * vec_a, dim=1)
    dc1 = torch.sum(vec_c * vec_b, dim=1)
    denom = d00 * d11 - d01 * d01
    valid_edges = denom > 1e-12
    u = (dc0 * d11 - dc1 * d01) / (denom + 1e-12)
    v = (d01 * dc0 - d00 * dc1) / (denom + 1e-12)
    inside = (u >= -eps) & (u <= 1.0 + eps) & (v >= -eps) & (v <= 1.0 + eps)

    is_colliding = found & valid_edges & inside & (t_final >= 0) & (t_final <= 1)
    t_final = torch.where(is_colliding, t_final, torch.tensor(-1.0, device=p0_start.device))
    return t_final, is_colliding
