"""Preprocessing branches and the end-to-end pipeline builder (M5, DOC-03 §7)."""

from house_price.pipelines.branches import build_column_transformer, check_group_coverage
from house_price.pipelines.build import build_pipeline

__all__ = ["build_column_transformer", "build_pipeline", "check_group_coverage"]
