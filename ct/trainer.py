"""Training loop (single GPU or DDP via torchrun)."""
import json
import math
import os
import random
import time
import warnings
from datetime import datetime

import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter

from ct.ccd import CCDLoss
from ct.data import ClothDataset, cloth_collate_fn, split_to_json
from ct.distributed import barrier
from ct.losses import contact_loss, linear_ramp, reconstruction_loss, self_contact_loss
from ct.rollout import DTYPES, autocast_context


# SequentialLR passes an epoch to its sub-schedulers internally, which PyTorch warns about.
warnings.filterwarnings("ignore", message="The epoch parameter in `scheduler.step()`")


class _NullWriter:
    """Stand-in for SummaryWriter in dry runs."""

    def __getattr__(self, _name):
        return lambda *args, **kwargs: None


class Trainer:
    def __init__(self, cfg, model, splits, rank, world_size, local_rank, device):
        self.cfg = cfg
        self.splits = splits
        self.rank, self.world_size, self.device = rank, world_size, device
        self.is_main = rank == 0
        t = cfg.training

        self.block = int(cfg.model.block_size)
        self.dt = float(cfg.data.dt)
        self.vel_scale = float(cfg.data.velocity_scale)
        self.n_steps = int(t.n_train_steps)
        self.dtype = DTYPES[cfg.runtime.dtype]
        self.dry_run = bool(cfg.runtime.dry_run)

        rc = t.rollout_curriculum
        assert int(rc.start_len) % self.block == 0 and int(rc.end_len) % self.block == 0, \
            "rollout lengths must be multiples of model.block_size"

        # Each sample holds the history block plus the longest rollout.
        seq_len = self.block + int(rc.end_len)
        make_set = lambda files, train: ClothDataset(
            files, seq_len, self.dt, self.vel_scale,
            augment=train and bool(t.augmentation.enabled),
            translate_range=float(t.augmentation.translate_range))
        self.train_set = make_set(splits["train"], True)
        self.val_set = make_set(splits["val"], False)
        if t.val_max_batches is not None:
            # Evenly spaced windows across all validation trajectories.
            n = min(len(self.val_set), int(t.val_max_batches) * int(t.batch_size) * world_size)
            idx = torch.linspace(0, len(self.val_set) - 1, n).long().unique().tolist() if n > 0 else []
            self.val_set = Subset(self.val_set, idx)
        self.train_sampler = DistributedSampler(self.train_set, num_replicas=world_size, rank=rank,
                                                shuffle=True, seed=int(t.seed), drop_last=True)
        val_sampler = DistributedSampler(self.val_set, num_replicas=world_size, rank=rank,
                                         shuffle=False, drop_last=False)
        loader = lambda ds, sampler: DataLoader(ds, batch_size=int(t.batch_size), sampler=sampler,
                                                num_workers=int(cfg.data.num_workers), pin_memory=True,
                                                collate_fn=cloth_collate_fn)
        self.train_loader = loader(self.train_set, self.train_sampler)
        self.val_loader = loader(self.val_set, val_sampler)
        print(f"[rank {rank}] {len(self.train_sampler)} training windows, {len(self.train_loader)} batches per epoch")

        self.model = model
        if world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[local_rank] if torch.cuda.is_available() else None,
                output_device=local_rank if torch.cuda.is_available() else None)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(t.lr),
                                           weight_decay=float(t.weight_decay), fused=False)
        self.scheduler = None
        if t.use_scheduler:
            self.scheduler = SequentialLR(self.optimizer, schedulers=[
                LinearLR(self.optimizer, start_factor=1.0, end_factor=1.0, total_iters=1),
                CosineAnnealingLR(self.optimizer, T_max=max(self.n_steps - 1, 1), eta_min=float(t.min_lr)),
            ], milestones=[1])

        c = cfg.loss.ccd
        self.ccd_loss = CCDLoss(margin=float(c.broad_phase_margin), safe_margin=float(c.safe_margin),
                                max_pairs=int(c.max_pairs), enable_self_vf=bool(c.self_vf),
                                enable_self_ee=bool(c.self_ee)) if c.enabled else None

        self.run_dir = os.path.join(cfg.runtime.output_dir,
                                    cfg.runtime.run_name or datetime.now().strftime("%Y%m%d_%H%M%S"))
        self.writer = None

    # ------------------------------------------------------------------ schedules
    def rollout_len(self, step: int) -> int:
        rc = self.cfg.training.rollout_curriculum
        if not rc.enabled or step >= int(rc.ramp_up_steps):
            return int(rc.end_len)
        if step < 0:
            return int(rc.start_len)
        length = int(rc.start_len) + (int(rc.end_len) - int(rc.start_len)) * step / int(rc.ramp_up_steps)
        return max(int(rc.start_len), int(length))

    def teacher_forcing_prob(self, step: int) -> float:
        ss = self.cfg.training.scheduled_sampling
        if not ss.enabled:
            return 0.0
        return linear_ramp(step, 0, self.n_steps, float(ss.start_prob), float(ss.end_prob))

    def contact_weight(self, step: int) -> float:
        c = self.cfg.loss.contact
        return linear_ramp(step, int(c.rampup_start), int(c.rampup_steps),
                           float(c.weight_start), float(c.weight_max))

    # ------------------------------------------------------------------ one step
    def run_step(self, batch, step: int, train: bool = True) -> dict:
        """Autoregressive multi-step loss on one batch; optimizer step if ``train``."""
        cfg, dev, block = self.cfg, self.device, self.block
        lc = cfg.loss
        self.model.train(train)
        traj = batch["traj"].to(dev, non_blocking=True)
        traj_vel = batch["traj_vel"].to(dev, non_blocking=True)
        collision = batch["collision"].to(dev, non_blocking=True)
        collision_vel = batch["collision_vel"].to(dev, non_blocking=True)
        rest_pos = batch["rest_pos"].to(dev, non_blocking=True)
        adj = batch["adj"].to(dev, non_blocking=True)
        cloth_mask = batch["cloth_mask"].to(dev, non_blocking=True)
        col_mask = batch["collision_mask"].to(dev, non_blocking=True)
        B, _, N_v, _ = traj.shape

        n_rollout = int(math.ceil((self.rollout_len(step) if train else
                                   int(cfg.training.rollout_curriculum.end_len)) / block)) * block
        n_in = block
        zero = lambda: torch.tensor(0.0, device=dev)
        total_loss, total_mse, real_max = zero(), zero(), zero()
        total_contact, total_ccd, total_self_contact = zero(), zero(), zero()
        per_step_mse = []

        with torch.enable_grad() if train else torch.no_grad():
            if train:
                self.optimizer.zero_grad(set_to_none=True)
            history = traj[:, :n_in].clone()
            history_vel = traj_vel[:, :n_in].clone()
            noise_std = float(cfg.training.noise_std)
            if train and noise_std > 0:
                history = history + torch.randn_like(history) * noise_std
                history_vel = history_vel + torch.randn_like(history_vel) * noise_std

            for k in range(0, n_rollout, block):
                frame = n_in + k  # first frame of the block being predicted
                col_in = collision[:, frame:frame + block]
                col_vel_in = collision_vel[:, frame:frame + block]
                with autocast_context(self.dtype, dev):
                    pred = self.model(history, col_in, history_vel, col_vel_in, rest_pos, adj,
                                      cloth_mask, col_mask)  # (B, n_in, N_v, 3)
                gt = traj[:, k + block:k + block + n_in]

                valid = cloth_mask.repeat(1, n_in).view(-1)
                loss, mse, step_max = reconstruction_loss(pred.reshape(-1, 3)[valid], gt.reshape(-1, 3)[valid])
                total_loss = total_loss + loss

                if train and lc.contact.enabled:
                    col_target = collision[:, k + block:k + block + n_in]
                    cl = zero()
                    for b in range(B):
                        for f in range(n_in):
                            cl = cl + contact_loss(
                                pred[b, f][cloth_mask[b]], col_target[b, f][col_mask[b]],
                                thresh=float(lc.contact.thresh), knn=int(lc.contact.knn),
                                max_dist=float(lc.contact.max_dist), depth_cap=float(lc.contact.depth_cap))
                    cl = cl / max(B * n_in, 1) * self.contact_weight(step)
                    total_loss = total_loss + cl
                    total_contact += cl.detach()

                if train and self.ccd_loss is not None:
                    # Sweep from the (penetration-free) ground-truth state to the prediction.
                    ccd_sum, counts = zero(), {"self_vf": 0, "self_ee": 0}
                    empty_v = torch.zeros(0, 3, device=dev)
                    empty_e = torch.zeros(0, 2, dtype=torch.long, device=dev)
                    empty_f = torch.zeros(0, 3, dtype=torch.long, device=dev)
                    for b in range(B):
                        n_vb = batch["n_verts"][b]
                        e_b = batch["edges"][b].to(dev)
                        f_b = batch["cloth_tris"][b].to(dev)
                        for f in range(n_in):
                            l, info = self.ccd_loss(gt[b, f, :n_vb], pred[b, f, :n_vb], empty_v, empty_v,
                                                    e_b, empty_e, empty_f, f_b,
                                                    n_vb, 0, e_b.shape[0], 0, 0, f_b.shape[0])
                            ccd_sum = ccd_sum + l
                            counts["self_vf"] += info["self_vf"]
                            counts["self_ee"] += info["self_ee"]
                    ccd = ccd_sum / max(B * n_in, 1) * float(lc.ccd.weight)
                    total_loss = total_loss + ccd
                    total_ccd += ccd.detach()
                    if self.is_main and (step % int(cfg.training.log_every) == 0 or step <= 5):
                        print(f"    [CCD loss] {ccd.item():.6e} (self_vf={counts['self_vf']}, "
                              f"self_ee={counts['self_ee']})")

                if train and lc.self_contact.enabled:
                    sc = zero()
                    for b in range(B):
                        n_vb = batch["n_verts"][b]
                        f_b = batch["cloth_tris"][b].to(dev)
                        for f in range(n_in):
                            sc = sc + self_contact_loss(pred[b, f, :n_vb], f_b,
                                                        thresh=float(lc.self_contact.thresh),
                                                        knn=int(lc.self_contact.knn))
                    sc = sc / max(B * n_in, 1) * float(lc.self_contact.weight)
                    total_loss = total_loss + sc
                    total_self_contact += sc.detach()

                total_mse += mse.detach()
                real_max = torch.max(real_max, step_max.detach())
                per_step_mse.append(mse.detach())

                # Next history block: model prediction (or ground truth for
                # scheduled sampling / validation).
                gt_next = traj[:, frame:frame + block]
                gt_next_vel = traj_vel[:, frame:frame + block]
                use_gt = not train or (self.teacher_forcing_prob(step) > 0
                                       and random.random() < self.teacher_forcing_prob(step))
                if use_gt:
                    next_pos, next_vel = gt_next, gt_next_vel
                else:
                    next_pos = pred[:, -block:].detach()
                    next_vel = torch.diff(torch.cat([history[:, -1:], next_pos], dim=1), dim=1) / self.dt * self.vel_scale
                    vel_noise = float(cfg.training.pred_vel_noise_std)
                    if vel_noise > 0:
                        next_vel = next_vel + torch.randn_like(next_vel) * vel_noise
                history = torch.cat([history[:, block:], next_pos], dim=1)
                history_vel = torch.cat([history_vel[:, block:], next_vel], dim=1)

            n_blocks = n_rollout // block
            avg_loss = total_loss / n_blocks
            if train:
                avg_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), float(cfg.training.grad_clip))
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        return {
            "loss": avg_loss.detach(),
            "mse": (total_mse / n_blocks).detach(),
            "max_sq_err": real_max.detach(),
            "contact": (total_contact / n_blocks).detach(),
            "ccd": (total_ccd / n_blocks).detach(),
            "self_contact": (total_self_contact / n_blocks).detach(),
            "per_step_mse": per_step_mse,
        }

    # ------------------------------------------------------------------ validation
    @torch.no_grad()
    def validate(self, step: int):
        start = time.time()
        loss_sum = torch.tensor(0.0, device=self.device)
        count = torch.tensor(0.0, device=self.device)
        for batch in self.val_loader:
            out = self.run_step(batch, step, train=False)
            bs = batch["traj"].shape[0]
            loss_sum += out["loss"] * bs
            count += bs
        if self.world_size > 1:
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count, op=dist.ReduceOp.SUM)
        val_loss = (loss_sum / count.clamp(min=1)).item()
        if self.is_main:
            print(f"Step {step}: val loss {val_loss:.6f} ({time.time() - start:.1f}s)")
            self.writer.add_scalar("loss/val", val_loss, step)

    # ------------------------------------------------------------------ main loop
    def save_checkpoint(self, step: int):
        path = os.path.join(self.run_dir, "checkpoints", f"step_{step:07d}.pt")
        if self.dry_run:
            print(f"[dry_run] skip saving {path}")
            return
        base = self.model.module if hasattr(self.model, "module") else self.model
        torch.save(base.state_dict(), path)
        print(f"Saved {path}")

    def fit(self):
        t = self.cfg.training
        if self.is_main:
            if self.dry_run:
                self.writer = _NullWriter()
                print("[dry_run] no TensorBoard logs or checkpoints will be written.")
            else:
                os.makedirs(os.path.join(self.run_dir, "checkpoints"), exist_ok=True)
                OmegaConf.save(self.cfg, os.path.join(self.run_dir, "config.yaml"))
                with open(os.path.join(self.run_dir, "split.json"), "w") as f:
                    json.dump(split_to_json(self.splits, self.cfg.data.dir), f, indent=1)
                self.writer = SummaryWriter(os.path.join(self.run_dir, "tb"))
                print(f"Run directory: {self.run_dir}")
            print(f"Training on {self.world_size} process(es), device={self.device}")

        step, epoch = 0, 0
        log_every, val_every, save_every = int(t.log_every), int(t.val_every), int(t.save_every)
        while step < self.n_steps:
            self.train_sampler.set_epoch(epoch)
            for batch in self.train_loader:
                step += 1
                out = self.run_step(batch, step, train=True)

                if self.is_main and (step % log_every == 0 or step == 1):
                    msg = f"[step {step}] loss {out['loss']:.6e}, mse {out['mse']:.6e}"
                    if self.cfg.loss.contact.enabled:
                        msg += f", contact {out['contact']:.6e} (w={self.contact_weight(step):.2e})"
                    if self.cfg.loss.ccd.enabled:
                        msg += f", ccd {out['ccd']:.6e}"
                    if self.cfg.loss.self_contact.enabled:
                        msg += f", self_contact {out['self_contact']:.6e}"
                    print(msg, flush=True)
                    w = self.writer
                    w.add_scalar("loss/train", out["loss"], step)
                    w.add_scalar("loss/train_mse", out["mse"], step)
                    w.add_scalar("loss/train_max_sq_err", out["max_sq_err"], step)
                    w.add_scalar("loss/contact", out["contact"], step)
                    w.add_scalar("loss/ccd", out["ccd"], step)
                    w.add_scalar("loss/self_contact", out["self_contact"], step)
                    w.add_scalar("lr", self.optimizer.param_groups[0]["lr"], step)
                    w.add_scalar("rollout/length", self.rollout_len(step), step)
                    for i, m in enumerate(out["per_step_mse"]):
                        w.add_scalar(f"loss_per_step/step_{i}", m.item(), step)

                if step % val_every == 0:
                    if len(self.val_loader) > 0:
                        self.validate(step)
                    barrier()

                if self.is_main and save_every > 0 and (step % save_every == 0 or step == self.n_steps):
                    self.save_checkpoint(step)

                if step >= self.n_steps:
                    break
            epoch += 1

        if self.is_main:
            print(f"Finished {self.n_steps} steps.")
            if self.writer is not None:
                self.writer.close()
