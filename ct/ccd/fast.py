"""Fully batched GPU collision primitives used by the CCD post-processor.

* ``aabb_pairs``: spatial-hash broad phase (no dense N x M matrix).
* ``vf_ccd`` / ``ee_ccd``: point-triangle / edge-edge continuous collision
  detection under linear motion over t in [0, 1]. The coplanarity cubic is
  sampled on a uniform grid and every sign-change interval is bisected in
  parallel, so all roots (not only the first one) are checked for a hit.
* closest-point helpers for the discrete (thickness) proximity test.
"""
import torch


class _MaybeCompiled:
    """``torch.compile`` (dynamic shapes) with a permanent fallback to eager mode."""

    def __init__(self, fn):
        self.fn, self.compiled, self.failed = fn, None, False

    def __call__(self, *args, **kwargs):
        if not self.failed:
            try:
                if self.compiled is None:
                    self.compiled = torch.compile(self.fn, dynamic=True)
                return self.compiled(*args, **kwargs)
            except Exception as e:  # e.g. no triton / unsupported platform
                print(f"[CCD] torch.compile unavailable ({type(e).__name__}); using eager mode.")
                self.failed = True
        return self.fn(*args, **kwargs)


# ----------------------------------------------------------------------------
# Broad phase
# ----------------------------------------------------------------------------

def _cells(lo, hi, cell):
    """Integer cells overlapped by each box. Returns (box index, cell key)."""
    c_lo = torch.floor(lo / cell).long()
    c_hi = torch.floor(hi / cell).long()
    span = c_hi - c_lo + 1
    n = span.prod(dim=1)
    box = torch.repeat_interleave(torch.arange(lo.shape[0], device=lo.device), n)
    start = torch.cumsum(n, 0) - n
    local = torch.arange(box.shape[0], device=lo.device) - start[box]
    sx, sy = span[box, 0], span[box, 1]
    ix, iy, iz = local % sx, (local // sx) % sy, local // (sx * sy)
    c = c_lo[box] + torch.stack([ix, iy, iz], dim=1)
    # Injective key for |cell index| < 2^20 per axis.
    c = c + (1 << 20)
    key = (c[:, 0] << 42) | (c[:, 1] << 21) | c[:, 2]
    return box, key


def aabb_pairs(lo_a, hi_a, lo_b, hi_b, cell: float = None, same: bool = False,
               max_cells_per_box: float = 8.0) -> torch.Tensor:
    """All (i, j) with overlapping boxes a_i, b_j. Returns (P, 2) long.

    ``cell`` defaults to the median box size; it is doubled until the boxes
    cover at most ``max_cells_per_box`` cells on average. With ``same=True``
    (a and b are the same set) only pairs with i < j are returned.
    """
    dev = lo_a.device
    empty = torch.zeros((0, 2), dtype=torch.long, device=dev)
    if lo_a.shape[0] == 0 or lo_b.shape[0] == 0:
        return empty
    if cell is None:
        ext = torch.cat([hi_a - lo_a, hi_b - lo_b]).amax(dim=1)
        cell = float(ext.median().clamp(min=1e-6))
    n_box = lo_a.shape[0] + lo_b.shape[0]
    while True:
        spans = torch.floor(torch.cat([hi_a, hi_b]) / cell) - torch.floor(torch.cat([lo_a, lo_b]) / cell) + 1
        if float(spans.prod(dim=1).sum()) <= max_cells_per_box * n_box:
            break
        cell *= 2
    box_a, key_a = _cells(lo_a, hi_a, cell)
    box_b, key_b = _cells(lo_b, hi_b, cell)
    key_b, order = torch.sort(key_b)
    box_b = box_b[order]
    left = torch.searchsorted(key_b, key_a, right=False)
    right = torch.searchsorted(key_b, key_a, right=True)
    cnt = right - left
    total = int(cnt.sum())
    if total == 0:
        return empty
    ia = torch.repeat_interleave(box_a, cnt)
    start = torch.cumsum(cnt, 0) - cnt
    pos = torch.repeat_interleave(left - start, cnt) + torch.arange(total, device=dev)
    ib = box_b[pos]
    keep = (lo_a[ia] <= hi_b[ib]).all(1) & (hi_a[ia] >= lo_b[ib]).all(1)
    if same:
        keep &= ia < ib
    ia, ib = ia[keep], ib[keep]
    if ia.numel() == 0:
        return empty
    uid = torch.unique(ia * lo_b.shape[0] + ib)
    return torch.stack([uid // lo_b.shape[0], uid % lo_b.shape[0]], dim=1)


def swept_box(*points_pairs, margin: float = 0.0):
    """AABB of the union of several (start, end) point sets of shape (N, 3)."""
    pts = torch.stack([p for pair in points_pairs for p in pair], dim=1)
    return pts.min(dim=1).values - margin, pts.max(dim=1).values + margin


# ----------------------------------------------------------------------------
# Narrow phase (continuous)
# ----------------------------------------------------------------------------

def _cubic_coeffs(m0_s, m0_v, m1_s, m1_v, m2_s, m2_v):
    """Coefficients (P, 4) of det[m0(t), m1(t), m2(t)] with m(t) = m_s + t m_v."""
    def tp(a, b, c):
        return (a * torch.cross(b, c, dim=-1)).sum(-1)
    a = tp(m0_v, m1_v, m2_v)
    b = tp(m0_s, m1_v, m2_v) + tp(m0_v, m1_s, m2_v) + tp(m0_v, m1_v, m2_s)
    c = tp(m0_v, m1_s, m2_s) + tp(m0_s, m1_v, m2_s) + tp(m0_s, m1_s, m2_v)
    d = tp(m0_s, m1_s, m2_s)
    return torch.stack([a, b, c, d], dim=-1)


def _roots(coeffs, n_samples: int, n_bisect: int):
    """Roots of the cubic in every sign-change sub-interval of [0, 1].

    Returns (t, valid) of shape (P, n_samples).
    """
    ts = torch.linspace(0, 1, n_samples + 1, device=coeffs.device, dtype=coeffs.dtype)
    a, b, c, d = coeffs.unbind(-1)
    f = lambda t: ((a[:, None] * t + b[:, None]) * t + c[:, None]) * t + d[:, None]
    fv = f(ts[None, :].expand(coeffs.shape[0], -1))
    valid = fv[:, :-1] * fv[:, 1:] <= 0
    lo = ts[:-1].expand(coeffs.shape[0], -1).clone()
    hi = ts[1:].expand(coeffs.shape[0], -1).clone()
    f_lo = fv[:, :-1]
    for _ in range(n_bisect):
        mid = 0.5 * (lo + hi)
        f_mid = f(mid)
        left = f_lo * f_mid <= 0
        hi = torch.where(left, mid, hi)
        lo = torch.where(left, lo, mid)
        f_lo = torch.where(left, f_lo, f_mid)
    return 0.5 * (lo + hi), valid


def _barycentric(p, a, b, c):
    v0, v1, v2 = b - a, c - a, p - a
    d00, d01, d11 = (v0 * v0).sum(-1), (v0 * v1).sum(-1), (v1 * v1).sum(-1)
    d20, d21 = (v2 * v0).sum(-1), (v2 * v1).sum(-1)
    denom = d00 * d11 - d01 * d01
    v = (d11 * d20 - d01 * d21) / (denom + 1e-20)
    w = (d00 * d21 - d01 * d20) / (denom + 1e-20)
    return torch.stack([1 - v - w, v, w], dim=-1), denom > 1e-20


def vf_ccd(p0, p1, a0, a1, b0, b1, c0, c1, n_samples: int = 8, n_bisect: int = 6, eps: float = 1e-6):
    """Point (p) vs triangle (a, b, c) CCD. Returns (hit (P,), toi (P,), 1 if no hit)."""
    coeffs = _cubic_coeffs(p0 - a0, (p1 - p0) - (a1 - a0), b0 - a0, (b1 - b0) - (a1 - a0),
                           c0 - a0, (c1 - c0) - (a1 - a0))
    t, valid = _roots(coeffs, n_samples, n_bisect)                    # (P, S)
    tt = t.unsqueeze(-1)
    lerp = lambda x0, x1: x0.unsqueeze(1) + tt * (x1 - x0).unsqueeze(1)
    bary, ok = _barycentric(lerp(p0, p1), lerp(a0, a1), lerp(b0, b1), lerp(c0, c1))
    valid = valid & ok & (bary >= -eps).all(-1)
    toi = torch.where(valid, t, torch.ones_like(t)).min(dim=1).values
    return valid.any(dim=1), toi


def _segment_params(p, q, r, s):
    """Closest points p + u (q - p), r + v (s - r) between segments. Returns u, v."""
    d1, d2, w = q - p, s - r, p - r
    a, e = (d1 * d1).sum(-1), (d2 * d2).sum(-1)
    b, c, f = (d1 * d2).sum(-1), (d1 * w).sum(-1), (d2 * w).sum(-1)
    denom = a * e - b * b
    u = torch.where(denom > 1e-20, (b * f - c * e) / (denom + 1e-20), torch.zeros_like(a)).clamp(0, 1)
    v = ((b * u + f) / (e + 1e-20)).clamp(0, 1)
    u = ((b * v - c) / (a + 1e-20)).clamp(0, 1)
    return u, v


def ee_ccd(p0, p1, q0, q1, r0, r1, s0, s1, n_samples: int = 8, n_bisect: int = 6, eps: float = 1e-6):
    """Edge (p, q) vs edge (r, s) CCD. Returns (hit (P,), toi (P,), 1 if no hit)."""
    coeffs = _cubic_coeffs(q0 - p0, (q1 - q0) - (p1 - p0), s0 - r0, (s1 - s0) - (r1 - r0),
                           r0 - p0, (r1 - r0) - (p1 - p0))
    t, valid = _roots(coeffs, n_samples, n_bisect)
    tt = t.unsqueeze(-1)
    lerp = lambda x0, x1: x0.unsqueeze(1) + tt * (x1 - x0).unsqueeze(1)
    P, Q, R, S = lerp(p0, p1), lerp(q0, q1), lerp(r0, r1), lerp(s0, s1)
    # The lines are coplanar at the root: intersect them (u on pq, v on rs).
    va, vb, vc = Q - P, S - R, R - P
    d00, d01, d11 = (va * va).sum(-1), (va * vb).sum(-1), (vb * vb).sum(-1)
    dc0, dc1 = (vc * va).sum(-1), (vc * vb).sum(-1)
    denom = d00 * d11 - d01 * d01
    u = (dc0 * d11 - dc1 * d01) / (denom + 1e-20)
    v = (d01 * dc0 - d00 * dc1) / (denom + 1e-20)
    valid = (valid & (denom > 1e-20) & (u >= -eps) & (u <= 1 + eps)
             & (v >= -eps) & (v <= 1 + eps))
    toi = torch.where(valid, t, torch.ones_like(t)).min(dim=1).values
    return valid.any(dim=1), toi


# ----------------------------------------------------------------------------
# Discrete proximity
# ----------------------------------------------------------------------------

def point_plane(p, a, b, c):
    """Unit normal of triangle (a, b, c), signed distance of p to its plane,
    barycentric coordinates of the projection and a non-degeneracy flag."""
    n = torch.cross(b - a, c - a, dim=-1)
    n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    d = ((p - a) * n).sum(-1)
    bary, ok = _barycentric(p - d.unsqueeze(-1) * n, a, b, c)
    return n, d, bary, ok


def segment_closest(p, q, r, s):
    """Closest points between segments (p, q) and (r, s): (u, v, vector from r-s point to p-q point)."""
    u, v = _segment_params(p, q, r, s)
    gap = (p + u.unsqueeze(-1) * (q - p)) - (r + v.unsqueeze(-1) * (s - r))
    return u, v, gap


vf_ccd_compiled = _MaybeCompiled(vf_ccd)
ee_ccd_compiled = _MaybeCompiled(ee_ccd)
