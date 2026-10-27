"""Inference-time CCD post-processing (non-differentiable, fully batched on GPU).

After every predicted frame, the motion of each cloth vertex from the previous
frame ``x0`` to the prediction ``x1`` is checked against the collider (moving
from ``c0`` to ``c1``) and against the cloth itself:

  VF       cloth vertex     vs collider triangle   (continuous + proximity within thickness)
  TV       collider vertex  vs cloth triangle      (continuous)
  EE       cloth edge       vs collider edge       (continuous)
  Self-VF  cloth vertex     vs cloth triangle      (continuous)
  Self-EE  cloth edge       vs cloth edge          (continuous)

Each iteration first resolves self contacts by moving the involved vertices back
along their motion to ``t_safe = max(0, t_c - self_time_margin)``, then pushes
the cloth to the side of the collider it came from, ``thickness`` away from the
surface (corrections of all contacts of a vertex are averaged). Candidate pairs
are found once per frame; later iterations only re-check pairs that contain a
vertex moved in the previous iteration.
"""
from dataclasses import dataclass, fields

import torch

from ct.ccd import fast
from ct.ccd.fast import aabb_pairs, point_plane, swept_box


@dataclass
class CCDParams:
    enabled: bool = True                # per-case switch (the global one is ccd_postprocess.enabled)
    max_iter: int = 5
    thickness: float = 0.002            # target cloth-collider distance (world units)
    self_time_margin: float = 0.2       # self contacts: step back this fraction of the time step
    relaxation: float = 1.0             # scale of the averaged cloth-collider correction
    max_depth: float = 0.01             # proximity test only within this depth below the surface
    broad_margin: float = 0.005         # candidate search margin (covers later corrections)
    collider_side: str = "start"        # "start": side at the previous frame; "outward": collider normals
    collider_vf: bool = True
    collider_tv: bool = True
    collider_ee: bool = True
    self_vf: bool = False               # self contacts are off by default
    self_ee: bool = False
    n_samples: int = 8                  # root search: samples of the coplanarity cubic ...
    n_bisect: int = 6                   # ... and bisection steps per sign change
    feedback: bool = True               # feed corrected frames back to the model (used by rollout)
    failsafe_iter: int = 0              # >0: roll back remaining crossings (this many partial passes, then to x0)
    failsafe_margin: float = 0.1        # roll back to (time of impact - margin) of the step
    compile: bool = True                # torch.compile the CCD tests (falls back to eager)

    @classmethod
    def from_config(cls, cfg):
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in dict(cfg).items() if k in names})


class CCDPostProcessor:
    def __init__(self, cloth_faces, cloth_edges, col_faces, col_edges, params: CCDParams):
        self.F, self.E = cloth_faces.long(), cloth_edges.long()
        self.CF, self.CE = col_faces.long(), col_edges.long()
        self.p = params
        self.vf_ccd = fast.vf_ccd_compiled if params.compile else fast.vf_ccd
        self.ee_ccd = fast.ee_ccd_compiled if params.compile else fast.ee_ccd

    # ------------------------------------------------------------ broad phase
    def _candidates(self, x0, x1, c0, c1):
        p, m = self.p, self.p.broad_margin
        F, E, CF, CE = self.F, self.E, self.CF, self.CE
        cand = {}
        if (p.self_vf or p.collider_tv) and F.shape[0]:
            tri_lo, tri_hi = swept_box((x0[F[:, 0]], x1[F[:, 0]]), (x0[F[:, 1]], x1[F[:, 1]]),
                                       (x0[F[:, 2]], x1[F[:, 2]]), margin=m)
        if (p.self_ee or p.collider_ee) and E.shape[0]:
            e_lo, e_hi = swept_box((x0[E[:, 0]], x1[E[:, 0]]), (x0[E[:, 1]], x1[E[:, 1]]), margin=m)
        v_lo, v_hi = swept_box((x0, x1), margin=m)
        self._vbox = (v_lo, v_hi)

        if p.collider_vf and CF.shape[0]:
            lo, hi = swept_box((c0[CF[:, 0]], c1[CF[:, 0]]), (c0[CF[:, 1]], c1[CF[:, 1]]),
                               (c0[CF[:, 2]], c1[CF[:, 2]]))
            pr = aabb_pairs(v_lo, v_hi, lo, hi)
            cand["vf"] = (pr[:, 0], CF[pr[:, 1]])
        if p.collider_tv and F.shape[0] and c0.shape[0]:
            lo, hi = swept_box((c0, c1))
            pr = aabb_pairs(lo, hi, tri_lo, tri_hi)
            cand["tv"] = (pr[:, 0], F[pr[:, 1]])
        if p.collider_ee and E.shape[0] and CE.shape[0]:
            lo, hi = swept_box((c0[CE[:, 0]], c1[CE[:, 0]]), (c0[CE[:, 1]], c1[CE[:, 1]]))
            pr = aabb_pairs(e_lo, e_hi, lo, hi)
            cand["ee"] = (E[pr[:, 0]], CE[pr[:, 1]])
        if p.self_vf and F.shape[0]:
            pr = aabb_pairs(v_lo, v_hi, tri_lo, tri_hi)
            v, f = pr[:, 0], F[pr[:, 1]]
            keep = (f != v.unsqueeze(-1)).all(-1)
            cand["svf"] = (v[keep], f[keep])
        if p.self_ee and E.shape[0]:
            pr = aabb_pairs(e_lo, e_hi, e_lo, e_hi, same=True)
            ea, eb = E[pr[:, 0]], E[pr[:, 1]]
            keep = (ea.unsqueeze(-1) != eb.unsqueeze(-2)).all(-1).all(-1)
            cand["see"] = (ea[keep], eb[keep])
        return cand

    # ---------------------------------------------------------- self contacts
    def _self_step(self, x0, x, cand, dirty):
        p = self.p
        t_safe = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
        n_hits = 0

        def clamp(verts, toi):
            ts = (toi - p.self_time_margin).clamp(min=0)
            for i in range(verts.shape[1]):
                t_safe.scatter_reduce_(0, verts[:, i], ts, reduce="amin")

        if "svf" in cand:
            v, f = cand["svf"]
            sel = dirty[v] | dirty[f].any(-1)
            v, f = v[sel], f[sel]
            if v.numel():
                hit, toi = self.vf_ccd(x0[v], x[v], x0[f[:, 0]], x[f[:, 0]], x0[f[:, 1]], x[f[:, 1]],
                                  x0[f[:, 2]], x[f[:, 2]], p.n_samples, p.n_bisect)
                if hit.any():
                    clamp(torch.cat([v[hit, None], f[hit]], dim=1), toi[hit])
                    n_hits += int(hit.sum())
        if "see" in cand:
            ea, eb = cand["see"]
            sel = dirty[ea].any(-1) | dirty[eb].any(-1)
            ea, eb = ea[sel], eb[sel]
            if ea.numel():
                hit, toi = self.ee_ccd(x0[ea[:, 0]], x[ea[:, 0]], x0[ea[:, 1]], x[ea[:, 1]],
                                  x0[eb[:, 0]], x[eb[:, 0]], x0[eb[:, 1]], x[eb[:, 1]],
                                  p.n_samples, p.n_bisect)
                if hit.any():
                    clamp(torch.cat([ea[hit], eb[hit]], dim=1), toi[hit])
                    n_hits += int(hit.sum())
        if n_hits:
            x = x0 + t_safe.unsqueeze(-1) * (x - x0)
        return x, n_hits

    # ------------------------------------------------------ cloth vs collider
    def _collider_setup(self, x0, c0, c1, cand):
        """Per-frame constants of the cloth-collider candidate pairs (collider and x0 are fixed)."""
        p, ctx = self.p, {}
        if "vf" in cand:
            v, f = cand["vf"]
            a0, b0, e0 = c0[f[:, 0]], c0[f[:, 1]], c0[f[:, 2]]
            a1, b1, e1 = c1[f[:, 0]], c1[f[:, 1]], c1[f[:, 2]]
            if p.collider_side == "outward":
                side = torch.ones(v.shape[0], device=x0.device, dtype=x0.dtype)
            else:
                _, d0, _, _ = point_plane(x0[v], a0, b0, e0)
                side = torch.where(d0 >= 0, 1.0, -1.0)
            ctx["vf"] = (v, x0[v], a0, a1, b0, b1, e0, e1, side)
        if "tv" in cand:
            k, f = cand["tv"]
            _, d0, _, _ = point_plane(c0[k], x0[f[:, 0]], x0[f[:, 1]], x0[f[:, 2]])
            ctx["tv"] = (f, c0[k], c1[k], torch.where(d0 >= 0, 1.0, -1.0))
        if "ee" in cand:
            e, ce = cand["ee"]
            pa, pb, qa, qb = x0[e[:, 0]], x0[e[:, 1]], c0[ce[:, 0]], c0[ce[:, 1]]
            # Separation direction at the previous frame, perpendicular to both edges.
            n = torch.cross(pb - pa, qb - qa, dim=-1)
            # Nearly parallel edges have no reliable separation direction (EE hits are
            # then covered by VF / TV of the edge end points).
            valid = n.norm(dim=-1) > 0.1 * (pb - pa).norm(dim=-1) * (qb - qa).norm(dim=-1)
            n = n / n.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            n = n * torch.where(((pa - qa) * n).sum(-1, keepdim=True) >= 0, 1.0, -1.0)
            ctx["ee"] = (e, pa, pb, qa, qb, c1[ce[:, 0]], c1[ce[:, 1]], n, valid)
        return ctx

    def _collider_step(self, x0, x, ctx):
        """One pass over the cloth-collider candidates. The contact types are applied
        one after another (VF, TV, EE); within a type, corrections of a vertex are
        averaged. Mask-based, without host synchronization."""
        p, h = self.p, self.p.thickness
        n_hits = torch.zeros((), device=x.device, dtype=torch.long)

        def apply(x, idx_list, disp_list, act):
            delta = torch.zeros_like(x)
            count = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
            for idx, disp in zip(idx_list, disp_list):
                delta.index_add_(0, idx, disp)
                count.index_add_(0, idx, act)
            return x + p.relaxation * delta / count.clamp(min=1).unsqueeze(-1)

        if "vf" in ctx:
            v, xv0, a0, a1, b0, b1, e0, e1, side = ctx["vf"]
            xv = x[v]
            hit, _ = self.vf_ccd(xv0, xv, a0, a1, b0, b1, e0, e1, p.n_samples, p.n_bisect)
            n1, d1, bary, ok = point_plane(xv, a1, b1, e1)
            sd = side * d1
            near = ok & (bary >= -1e-3).all(-1) & (sd < 0.99 * h) & (sd > -p.max_depth)
            act = (hit | near).to(x.dtype)
            x = apply(x, [v], [n1 * ((side * h - d1) * act).unsqueeze(-1)], act)
            n_hits += (act > 0).sum()

        if "tv" in ctx:
            f, pk0, pk1, side = ctx["tv"]
            t0, t1, t2 = x[f[:, 0]], x[f[:, 1]], x[f[:, 2]]
            hit, _ = self.vf_ccd(pk0, pk1, x0[f[:, 0]], t0, x0[f[:, 1]], t1, x0[f[:, 2]], t2,
                            p.n_samples, p.n_bisect)
            n1, d1, bary, _ = point_plane(pk1, t0, t1, t2)
            # Shift the cloth triangle along -n1 so the collider vertex ends up
            # `thickness` away on its original side.
            act = hit.to(x.dtype)
            s = (side * h - d1) * act
            w = bary.clamp(0, 1)
            w = w / (w * w).sum(-1, keepdim=True).clamp(min=1e-12)
            x = apply(x, [f[:, i] for i in range(3)],
                      [-n1 * (s * w[:, i]).unsqueeze(-1) for i in range(3)], act)
            n_hits += hit.sum()

        if "ee" in ctx:
            e, pa, pb, qa, qb, qa1, qb1, n, valid = ctx["ee"]
            ra, rb = x[e[:, 0]], x[e[:, 1]]
            hit, _ = self.ee_ccd(pa, ra, pb, rb, qa, qa1, qb, qb1, p.n_samples, p.n_bisect)
            hit = hit & valid
            act = hit.to(x.dtype)
            # n is perpendicular to both edges: any pair of points gives the distance.
            sd1 = (((ra + rb) / 2 - qa1) * n).sum(-1)
            s = (h - sd1).clamp(min=0) * act
            x = apply(x, [e[:, 0], e[:, 1]], [n * s.unsqueeze(-1)] * 2, act)
            n_hits += hit.sum()

        return x, n_hits

    # --------------------------------------------------------------- failsafe
    def _failsafe(self, x0, x, c0, c1):
        """Roll back the cloth vertices of remaining cloth-collider crossings along their
        motion: ``failsafe_iter`` passes to just before the time of impact, then to x0
        until no crossing is left. Rolling back a vertex changes the motion of its
        neighbours, so every pass re-checks all candidates."""
        p = self.p
        # Candidates of the push stage stay valid if no vertex left its search box;
        # rollbacks only shrink the motion x0 -> x.
        v_lo, v_hi = self._vbox
        if bool(((x >= v_lo) & (x <= v_hi)).all()):
            cand = self._cand
        else:
            cand = self._candidates(x0, x, c0, c1)
        for k in range(p.failsafe_iter + 32):
            to_start = k >= p.failsafe_iter
            t_safe = torch.ones(x.shape[0], device=x.device, dtype=x.dtype)
            n = torch.zeros((), device=x.device, dtype=torch.long)

            def clamp(verts, hit, toi):
                ts = torch.zeros_like(toi) if to_start else (toi - p.failsafe_margin).clamp(min=0)
                ts = torch.where(hit, ts, torch.ones_like(ts))
                for i in range(verts.shape[1]):
                    t_safe.scatter_reduce_(0, verts[:, i], ts, reduce="amin")

            if "vf" in cand:
                v, f = cand["vf"]
                hit, toi = self.vf_ccd(x0[v], x[v], c0[f[:, 0]], c1[f[:, 0]], c0[f[:, 1]], c1[f[:, 1]],
                                  c0[f[:, 2]], c1[f[:, 2]], p.n_samples, p.n_bisect)
                clamp(v[:, None], hit, toi)
                n += hit.sum()
            if "tv" in cand:
                kk, f = cand["tv"]
                hit, toi = self.vf_ccd(c0[kk], c1[kk], x0[f[:, 0]], x[f[:, 0]], x0[f[:, 1]], x[f[:, 1]],
                                  x0[f[:, 2]], x[f[:, 2]], p.n_samples, p.n_bisect)
                clamp(f, hit, toi)
                n += hit.sum()
            if "ee" in cand:
                e, ce = cand["ee"]
                hit, toi = self.ee_ccd(x0[e[:, 0]], x[e[:, 0]], x0[e[:, 1]], x[e[:, 1]],
                                  c0[ce[:, 0]], c1[ce[:, 0]], c0[ce[:, 1]], c1[ce[:, 1]],
                                  p.n_samples, p.n_bisect)
                clamp(e, hit, toi)
                n += hit.sum()
            if int(n) == 0:
                break
            x = x0 + t_safe.unsqueeze(-1) * (x - x0)
        return x

    # ------------------------------------------------------------------- main
    @torch.no_grad()
    def __call__(self, x0, x1, c0, c1):
        """Correct the predicted cloth positions ``x1`` (N, 3).

        Returns (corrected positions, iterations used, contacts found in the first iteration).
        """
        cand = self._cand = self._candidates(x0, x1, c0, c1)
        ctx = self._collider_setup(x0, c0, c1, cand)
        self_on = "svf" in cand or "see" in cand
        x = x1.clone()
        dirty = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
        first_hits, it = None, 0
        for it in range(1, self.p.max_iter + 1):
            n_self = 0
            if self_on:
                x_prev = x
                x, n_self = self._self_step(x0, x, cand, dirty)
                dirty = (x != x_prev).any(-1) | dirty
            x_prev = x
            x, n_col = self._collider_step(x0, x, ctx)
            n = n_self + int(n_col)
            if first_hits is None:
                first_hits = n
            if n == 0:
                break
            dirty = (x != x_prev).any(-1)
        if self.p.failsafe_iter > 0:
            x = self._failsafe(x0, x, c0, c1)
        return x, it, first_hits or 0
