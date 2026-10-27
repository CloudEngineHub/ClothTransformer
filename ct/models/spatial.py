"""
Spatial encoder/decoder for cloth simulation.

The latent-set compression follows 3DShape2VecSet (Zhang et al., SIGGRAPH 2023,
https://github.com/1zb/3DShape2VecSet), adapted for cloth meshes: learnable
latent queries compress cloth vertices and collider triangles into a fixed-size
latent set, rest-shape vertices query the latents back, and GNN blocks on both
sides propagate features along mesh edges.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ==================== Basic Components ==================== #

class GEGLU(nn.Module):
    """Gated GELU activation."""
    def forward(self, x):
        x, gates = x.chunk(2, dim=-1)
        return x * F.gelu(gates)


class FeedForward(nn.Module):
    """Feed-forward network with GEGLU activation."""
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Dropout(dropout),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class PointEmbed(nn.Module):
    """
    Sinusoidal positional embedding for 3D points.

    Encodes each (x, y, z) coordinate using sin/cos at multiple frequencies,
    similar to NeRF positional encoding but with axis-specific bands.

    As in 3DShape2VecSet:
    1. Create frequency bands for each axis (x, y, z)
    2. Project input coordinates onto frequency bands
    3. Apply sin/cos to get embeddings
    4. Concatenate with original coordinates
    5. Project to output dimension
    """
    def __init__(self, hidden_dim=48, dim=512):
        super().__init__()
        assert hidden_dim % 6 == 0, "hidden_dim must be divisible by 6"

        self.embedding_dim = hidden_dim
        num_freqs = self.embedding_dim // 6  # Number of frequency bands per axis

        # Create frequency bands: 2^0, 2^1, ..., 2^(num_freqs - 1)
        e = torch.pow(2, torch.arange(num_freqs)).float() * np.pi

        # Stack for x, y, z axes (each axis has its own frequency bands)
        # Each row has 3 * num_freqs elements = hidden_dim / 2 elements
        e = torch.stack([
            torch.cat([e, torch.zeros(num_freqs), torch.zeros(num_freqs)]),
            torch.cat([torch.zeros(num_freqs), e, torch.zeros(num_freqs)]),
            torch.cat([torch.zeros(num_freqs), torch.zeros(num_freqs), e]),
        ])
        self.register_buffer('basis', e)  # (3, hidden_dim / 2)

        # After sin/cos concatenation: hidden_dim dimensions
        # Plus original xyz: 3 dimensions
        # Total input to MLP: hidden_dim + 3
        self.mlp = nn.Linear(self.embedding_dim + 3, dim)

    def embed(self, input):
        """Apply sinusoidal embedding."""
        # input: (B, N, 3), basis: (3, hidden_dim / 2) -> projections: (B, N, hidden_dim / 2).
        # Computed element-wise in fp32 (each basis column has a single non-zero
        # entry): the highest frequencies (2^k * pi) need full precision, which
        # TF32 / half-precision matmuls under autocast would not provide.
        projections = (input.float().unsqueeze(-1) * self.basis).sum(dim=-2)
        # embeddings: (B, N, hidden_dim) after sin/cos concat
        embeddings = torch.cat([projections.sin(), projections.cos()], dim=-1)
        return embeddings

    def forward(self, input):
        """
        Args:
            input: (B, N, 3) - 3D point coordinates
        Returns:
            embeddings: (B, N, dim)
        """
        embed = self.embed(input)  # (B, N, hidden_dim)
        out = self.mlp(torch.cat([embed, input], dim=-1))  # (B, N, dim)
        return out


class CrossAttention(nn.Module):
    """
    Cross-attention module.

    Query attends to context (key/value from different source).
    """
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        context_dim = context_dim or query_dim
        inner_dim = dim_head * heads

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        """
        Args:
            x: (B, N, D) - query
            context: (B, M, D) - key/value source (if None, self-attention)
            mask: (B, M) - True for valid positions
        Returns:
            out: (B, N, D)
        """
        h = self.heads

        q = self.to_q(x)
        context = context if context is not None else x
        k, v = self.to_kv(context).chunk(2, dim=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))
        if mask is not None:
            mask = mask[:, None, None, :]  # (B, 1, 1, M), True = attend
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=self.scale)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.to_out(out)


class SelfAttention(nn.Module):
    """Self-attention module."""
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):
        super().__init__()
        self.cross_attn = CrossAttention(dim, dim, heads, dim_head, dropout)

    def forward(self, x, mask=None):
        return self.cross_attn(x, context=x, mask=mask)


# ==================== GNN Components ==================== #

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution Layer with message passing.

    Aggregates features from neighboring vertices using the adjacency matrix.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.act = nn.GELU()

    def forward(self, x, adj_matrix):
        """
        Args:
            x: (B, N, C) - node features
            adj_matrix: (B, N, N) - row-normalized adjacency matrix
        Returns:
            out: (B, N, C)
        """
        # Feature transformation
        out = self.linear(x)

        # Message passing: aggregate neighbor features
        out = torch.bmm(adj_matrix, out)

        return self.act(out)


class GNNBlock(nn.Module):
    """
    Multi-layer GNN block with residual connections.
    """
    def __init__(self, dim, num_layers=2, dropout=0.0):
        super().__init__()
        self.layers = nn.ModuleList([
            GraphConvLayer(dim, dim) for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([
            nn.LayerNorm(dim) for _ in range(num_layers)
        ])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, adj_matrix):
        """
        Args:
            x: (B, N, D)
            adj_matrix: (B, N, N)
        Returns:
            out: (B, N, D)
        """
        for layer, norm in zip(self.layers, self.norms):
            residual = x
            x = norm(x)
            x = layer(x, adj_matrix)
            x = self.dropout(x)
            x = x + residual
        return x


# ==================== Latent-Query Decoder ==================== #

class LatentQueryDecoder(nn.Module):
    """
    Decoder in which every query vertex attends to the latent set.

    Architecture:
    1. Cross-Attention: Each vertex queries the latent vectors
    2. GNN: Refine predictions using mesh topology
    3. Output projection: Project to output dimension
    """
    def __init__(
        self,
        dim=512,
        output_dim=3,
        point_embed_dim=48,
        heads=8,
        dim_head=64,
        gnn_layers=2,
        ff_mult=4,
        dropout=0.0,
    ):
        super().__init__()

        self.dim = dim

        # Point embedding for query positions
        self.point_embed = PointEmbed(hidden_dim=point_embed_dim, dim=dim)

        # Cross-attention: vertices query latents
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.cross_attn_norm_q = nn.LayerNorm(dim)
        self.cross_attn_norm_kv = nn.LayerNorm(dim)

        # Feedforward
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)

        # GNN for topology-aware refinement
        self.gnn = GNNBlock(dim, num_layers=gnn_layers, dropout=dropout)

        # Output projection
        self.to_output = nn.Linear(dim, output_dim)

    def forward(self, latents, query_positions, adj_matrix=None):
        """
        Decode latent vectors to vertex predictions.

        Args:
            latents: (B, M, D) - latent vectors from encoder
            query_positions: (B, N, 3) - positions to query (e.g., rest positions)
            adj_matrix: (B, N, N) - normalized adjacency matrix (optional)

        Returns:
            output: (B, N, output_dim) - predicted values at each vertex
        """
        B, N, _ = query_positions.shape

        # 1. Embed query positions
        queries = self.point_embed(query_positions)  # (B, N, D)

        # 2. Cross-attention: vertices query latents
        queries_normed = self.cross_attn_norm_q(queries)
        latents_normed = self.cross_attn_norm_kv(latents)
        x = self.cross_attn(queries_normed, context=latents_normed) + queries

        # 3. Feedforward
        x = self.ff(self.ff_norm(x)) + x

        # 4. GNN refinement
        if adj_matrix is not None:
            x = self.gnn(x, adj_matrix)

        # 5. Output projection
        output = self.to_output(x)

        return output


# ==================== Combined Encoder for Cloth + Collision ==================== #

class SpatialEncoder(nn.Module):
    """
    Combined spatial encoder for cloth vertices and collision geometry.

    Encodes both cloth mesh and collision triangles into a unified latent space
    for temporal processing.
    """
    def __init__(
        self,
        dim=512,
        num_latents=128,
        point_embed_dim=48,
        heads=8,
        dim_head=64,
        gnn_layers=2,
        num_self_attn_layers=0,
        ff_mult=4,
        dropout=0.0,
    ):
        super().__init__()

        self.dim = dim
        self.num_latents = num_latents

        # Cloth encoder
        self.cloth_point_embed = PointEmbed(hidden_dim=point_embed_dim, dim=dim)
        
        # Velocity embedding for cloth
        self.cloth_vel_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )
        self.cloth_fusion = nn.Linear(dim * 2, dim) # Fuse pos_emb + vel_emb

        self.cloth_gnn = GNNBlock(dim, num_layers=gnn_layers, dropout=dropout)

        # Collision encoder: PointEmbed of each triangle corner + explicit
        # geometric features (unit normal and area).
        self.collision_point_embed = PointEmbed(hidden_dim=point_embed_dim, dim=dim)
        
        # Velocity embedding for collision
        self.collision_vel_embed = nn.Sequential(
            nn.Linear(3, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

        # Geometric features: normal (3D) + area (1D) = 4D
        self.collision_geo_proj = nn.Sequential(
            nn.Linear(4, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )
        # Combine vertex embeddings + geometric features
        self.collision_combine = nn.Sequential(
            nn.Linear(dim * 3, dim),  # (vertex_emb + vel_emb + geo_emb) -> dim
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        # Token type embedding to distinguish cloth vs collision
        self.token_type_embed = nn.Embedding(2, dim)

        # Learnable latent queries
        self.latent_queries = nn.Parameter(torch.randn(num_latents, dim))

        # Cross-attention
        self.cross_attn = CrossAttention(
            query_dim=dim,
            context_dim=dim,
            heads=heads,
            dim_head=dim_head,
            dropout=dropout
        )
        self.cross_attn_norm = nn.LayerNorm(dim)

        # Feedforward
        self.ff = FeedForward(dim, mult=ff_mult, dropout=dropout)
        self.ff_norm = nn.LayerNorm(dim)

        # Self-attention layers on latent set (after cross-attention)
        self.self_attn_layers = nn.ModuleList()
        for _ in range(num_self_attn_layers):
            self.self_attn_layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                SelfAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                nn.LayerNorm(dim),
                FeedForward(dim, mult=ff_mult, dropout=dropout),
            ]))

    def forward(
        self,
        cloth_verts,
        collision_tris,
        cloth_vel,
        collision_vel,
        adj_matrix=None,
        cloth_padding_mask=None,
        collision_padding_mask=None,
    ):
        """
        Encode cloth vertices and collision geometry.

        Args:
            cloth_verts: (B, N_v, 3) - cloth vertex positions
            collision_tris: (B, N_t, 9) - collision triangle data
            cloth_vel: (B, N_v, 3) - cloth vertex velocities
            collision_vel: (B, N_t, 9) - collision triangle velocities
            adj_matrix: (B, N_v, N_v) - cloth mesh adjacency
            cloth_padding_mask: (B, N_v) - True for valid cloth vertices
            collision_padding_mask: (B, N_t) - True for valid collision triangles

        Returns:
            latents: (B, M, D) - compressed latent vectors
        """
        B = cloth_verts.shape[0]
        device = cloth_verts.device

        # 1. Encode cloth vertices
        cloth_pos_emb = self.cloth_point_embed(cloth_verts)  # (B, N_v, D)
        cloth_vel_emb = self.cloth_vel_embed(cloth_vel)      # (B, N_v, D)
        
        cloth_emb = self.cloth_fusion(torch.cat([cloth_pos_emb, cloth_vel_emb], dim=-1))

        if adj_matrix is not None:
            cloth_emb = self.cloth_gnn(cloth_emb, adj_matrix)
        cloth_emb = cloth_emb + self.token_type_embed(
            torch.zeros(1, dtype=torch.long, device=device)
        )

        # 2. Encode collision triangles with PointEmbed + explicit geometry
        # Extract vertices from triangle data (B, N_t, 9) -> 3 x (B, N_t, 3)
        v0 = collision_tris[..., 0:3]
        v1 = collision_tris[..., 3:6]
        v2 = collision_tris[..., 6:9]

        # Extract velocities
        vel0 = collision_vel[..., 0:3]
        vel1 = collision_vel[..., 3:6]
        vel2 = collision_vel[..., 6:9]

        # 2a. PointEmbed for each vertex, then average
        emb_v0 = self.collision_point_embed(v0)  # (B, N_t, D)
        emb_v1 = self.collision_point_embed(v1)
        emb_v2 = self.collision_point_embed(v2)
        vertex_emb = (emb_v0 + emb_v1 + emb_v2) / 3.0  # Average pooling

        # Velocity Embed
        emb_vel0 = self.collision_vel_embed(vel0)
        emb_vel1 = self.collision_vel_embed(vel1)
        emb_vel2 = self.collision_vel_embed(vel2)
        vertex_vel_emb = (emb_vel0 + emb_vel1 + emb_vel2) / 3.0

        # 2b. Compute explicit geometric features
        # Edge vectors
        edge1 = v1 - v0
        edge2 = v2 - v0
        # Normal vector (cross product, then normalize)
        normal = torch.cross(edge1, edge2, dim=-1)
        normal_length = torch.norm(normal, dim=-1, keepdim=True).clamp(min=1e-8)
        normal_normalized = normal / normal_length
        # Area = 0.5 * |normal| (before normalization)
        area = normal_length / 2.0

        # Concatenate geometric features: normal (3D) + area (1D) = 4D
        geo_features = torch.cat([normal_normalized, area], dim=-1)  # (B, N_t, 4)
        geo_emb = self.collision_geo_proj(geo_features)  # (B, N_t, D)

        # 2c. Combine vertex embedding and geometric embedding
        collision_emb = self.collision_combine(
            torch.cat([vertex_emb, vertex_vel_emb, geo_emb], dim=-1)
        )  # (B, N_t, D)

        # Add token type embedding
        collision_emb = collision_emb + self.token_type_embed(
            torch.ones(1, dtype=torch.long, device=device)
        )

        # 3. Concatenate cloth and collision
        combined = torch.cat([cloth_emb, collision_emb], dim=1)  # (B, N_v + N_t, D)

        # 4. Combined padding mask
        combined_mask = None
        if cloth_padding_mask is not None and collision_padding_mask is not None:
            combined_mask = torch.cat([cloth_padding_mask, collision_padding_mask], dim=1)

        # 5. Cross-attention compression
        queries = self.latent_queries.unsqueeze(0).expand(B, -1, -1)
        queries_normed = self.cross_attn_norm(queries)
        latents = self.cross_attn(queries_normed, context=combined, mask=combined_mask) + queries

        # 6. Feedforward
        latents = self.ff(self.ff_norm(latents)) + latents

        # 7. Self-attention layers on latent set
        for attn_norm, self_attn, ff_norm, ff in self.self_attn_layers:
            latents = self_attn(attn_norm(latents)) + latents
            latents = ff(ff_norm(latents)) + latents

        return latents


class SpatialDecoder(nn.Module):
    """
    Spatial decoder for cloth vertices.

    Decodes latent vectors back to cloth vertex positions.
    """
    def __init__(
        self,
        dim=512,
        output_dim=3,
        point_embed_dim=48,
        heads=8,
        dim_head=64,
        gnn_layers=2,
        ff_mult=4,
        dropout=0.0,
    ):
        super().__init__()

        self.decoder = LatentQueryDecoder(
            dim=dim,
            output_dim=output_dim,
            point_embed_dim=point_embed_dim,
            heads=heads,
            dim_head=dim_head,
            gnn_layers=gnn_layers,
            ff_mult=ff_mult,
            dropout=dropout,
        )

    def forward(self, latents, rest_positions, adj_matrix=None):
        """
        Decode latents to cloth vertex positions.

        Args:
            latents: (B, M, D) - latent vectors
            rest_positions: (B, N_v, 3) - rest pose positions for queries
            adj_matrix: (B, N_v, N_v) - mesh adjacency

        Returns:
            positions: (B, N_v, 3) - predicted vertex positions
        """
        return self.decoder(latents, rest_positions, adj_matrix)


# ==================== Factory Functions ==================== #

def build_spatial_encoder(
    dim=512,
    num_latents=128,
    gnn_layers=2,
    num_self_attn_layers=0,
    heads=8,
    dropout=0.0,
):
    """Build the spatial encoder (cloth + collider -> latent set)."""
    return SpatialEncoder(
        dim=dim,
        num_latents=num_latents,
        point_embed_dim=48,
        heads=heads,
        dim_head=dim // heads,
        gnn_layers=gnn_layers,
        num_self_attn_layers=num_self_attn_layers,
        ff_mult=4,
        dropout=dropout,
    )


def build_spatial_decoder(
    dim=512,
    output_dim=3,
    gnn_layers=2,
    heads=8,
    dropout=0.0,
):
    """Build the spatial decoder (latent set -> cloth vertex positions)."""
    return SpatialDecoder(
        dim=dim,
        output_dim=output_dim,
        point_embed_dim=48,
        heads=heads,
        dim_head=dim // heads,
        gnn_layers=gnn_layers,
        ff_mult=4,
        dropout=dropout,
    )
