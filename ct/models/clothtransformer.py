"""ClothTransformer: block-causal temporal Transformer over a latent-set representation.

Architecture (per frame, then across frames):
  1. Spatial encoder: cloth vertices (position + velocity, GNN) and lookahead
     collider triangles are compressed into ``num_latents`` latent tokens by
     cross-attention from learnable queries.
  2. Temporal Transformer: the latent sequence of all history frames is
     processed with block-causal attention (a block of ``block_size`` frames can
     attend to itself and to earlier blocks only).
  3. Spatial decoder: rest-shape vertices query the predicted latents, followed
     by GNN refinement and a linear projection to 3D positions.
"""
import math

import torch
import torch.nn as nn

from ct.models import register
from ct.models.attention import TransformerEncoder
from ct.models.spatial import build_spatial_decoder, build_spatial_encoder


class SinusoidalFrameEncoding(nn.Module):
    """Fixed sin/cos encoding of the frame index."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, frame_indices: torch.Tensor) -> torch.Tensor:
        return self.pe[frame_indices]


@register("clothtransformer")
class ClothTransformer(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            cfg: the ``model`` section of the config (see configs/default.yaml).
        """
        super().__init__()
        self.num_latents = int(cfg.num_latents)
        self.block_size = int(cfg.block_size)
        self.embed_dim = int(cfg.embed_dim)
        self.out_dim = 3
        dropout = float(cfg.dropout)

        self.spatial_encoder = build_spatial_encoder(
            dim=self.embed_dim,
            num_latents=self.num_latents,
            gnn_layers=int(cfg.spatial_gnn_layers),
            num_self_attn_layers=int(cfg.spatial_self_attn_layers),
            heads=int(cfg.num_heads),
            dropout=dropout,
        )
        self.spatial_decoder = build_spatial_decoder(
            dim=self.embed_dim,
            output_dim=self.out_dim,
            gnn_layers=int(cfg.spatial_gnn_layers),
            heads=int(cfg.num_heads),
            dropout=dropout,
        )

        # The model sees `block_size` history frames; the extra slots keep the
        # embedding table shape compatible with released checkpoints.
        max_frames = 2 * self.block_size + 1
        self.frame_pe = str(cfg.frame_pe)
        if self.frame_pe == "learnable":
            self.frame_pos_emb = nn.Embedding(max_frames, self.embed_dim)
        elif self.frame_pe == "sincos":
            self.frame_pos_emb = SinusoidalFrameEncoding(self.embed_dim, max_len=max_frames)
        elif self.frame_pe != "none":
            raise ValueError(f"model.frame_pe must be learnable | sincos | none, got {self.frame_pe!r}")

        self.encoder = TransformerEncoder(
            num_layers=int(cfg.num_layers),
            num_heads=int(cfg.num_heads),
            hidden_dim=self.embed_dim,
            ffn_hidden_dim=int(cfg.ffn_dim),
            activation='swiglu',
            dropout=dropout,
        )
        self._mask_cache_key = None
        self._mask_cache = None

    def _block_causal_mask(self, n_frames: int, device) -> torch.Tensor:
        """(n_frames*K, n_frames*K) bool mask, True = attention allowed."""
        key = (n_frames, device)
        if self._mask_cache_key != key:
            tokens_per_block = self.block_size * self.num_latents
            block_id = torch.arange(n_frames * self.num_latents, device=device) // tokens_per_block
            self._mask_cache = block_id.unsqueeze(1) >= block_id.unsqueeze(0)
            self._mask_cache_key = key
        return self._mask_cache

    def forward(self, src_verts, collision_hist, src_vel, collision_vel, rest_pos, adj_matrix=None,
                cloth_padding_mask=None, collision_padding_mask=None):
        """
        Args:
            src_verts: (B, T, N_v, 3) cloth vertex positions of the history frames
            collision_hist: (B, T, N_tris, 9) collider triangles, shifted one block ahead
            src_vel: (B, T, N_v, 3) scaled cloth vertex velocities
            collision_vel: (B, T, N_tris, 9) scaled collider triangle velocities
            rest_pos: (B, N_v, 3) rest-shape vertex positions
            adj_matrix: (B, N_v, N_v) row-normalized mesh adjacency
            cloth_padding_mask: (B, N_v) True for valid vertices
            collision_padding_mask: (B, N_tris) True for valid triangles

        Returns:
            (B, T, N_v, 3) predicted positions; entry t is the prediction for frame
            t + block_size.
        """
        B, T, N_v, _ = src_verts.shape
        N_tris = collision_hist.shape[2]
        device = src_verts.device
        if T % self.block_size != 0:
            raise ValueError(f"Number of input frames {T} is not a multiple of block_size {self.block_size}.")

        # 1. Spatial encoding, every frame independently.
        adj_flat = adj_matrix.repeat_interleave(T, dim=0) if adj_matrix is not None else None
        cloth_mask_flat = (cloth_padding_mask.repeat_interleave(T, dim=0)
                           if cloth_padding_mask is not None else None)
        collision_mask_flat = (collision_padding_mask.repeat_interleave(T, dim=0)
                               if collision_padding_mask is not None else None)
        latents = self.spatial_encoder(
            cloth_verts=src_verts.reshape(B * T, N_v, 3),
            collision_tris=collision_hist.reshape(B * T, N_tris, 9),
            cloth_vel=src_vel.reshape(B * T, N_v, 3),
            collision_vel=collision_vel.reshape(B * T, N_tris, 9),
            adj_matrix=adj_flat,
            cloth_padding_mask=cloth_mask_flat,
            collision_padding_mask=collision_mask_flat,
        )  # (B*T, K, D)

        # 2. Temporal Transformer over the latent sequence.
        latents_seq = latents.reshape(B, T * self.num_latents, self.embed_dim)
        if self.frame_pe != "none":
            frame_idx = torch.arange(T, device=device).repeat_interleave(self.num_latents)
            latents_seq = latents_seq + self.frame_pos_emb(frame_idx).unsqueeze(0)
        # With a single block every token may attend to every other one; skipping
        # the (all-True) mask lets SDPA use its fastest kernels.
        mask = self._block_causal_mask(T, device) if T > self.block_size else None
        latents_seq = self.encoder(latents_seq, attn_mask=mask)

        # 3. Spatial decoding, queried by the rest shape.
        preds = self.spatial_decoder(
            latents=latents_seq.reshape(B * T, self.num_latents, self.embed_dim),
            rest_positions=rest_pos.repeat_interleave(T, dim=0),
            adj_matrix=adj_flat,
        )
        return preds.reshape(B, T, N_v, self.out_dim)
