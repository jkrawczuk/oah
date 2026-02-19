"""
Online builder for hyperplane-based binary classification (OAH rules).

Expose the OnlineAdditiveHyperplanes as the main entrypoint.
"""

from .oah import CellStats, Hyperplane, OnlineAdditiveHyperplanes

__all__ = ["Hyperplane", "OnlineAdditiveHyperplanes", "CellStats"]
