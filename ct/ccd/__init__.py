"""Continuous collision detection: primitives, differentiable loss, post-processing."""
from .loss import CCDLoss
from .postprocess import CCDParams, CCDPostProcessor
from .primitives import edge_edge_ccd, point_triangle_ccd

__all__ = ["CCDLoss", "CCDParams", "CCDPostProcessor", "edge_edge_ccd", "point_triangle_ccd"]
