"""NSCF treatment-partition discovery from a pretrained neural outcome model."""
from .estimator import NSCF, make_anchor, train_anchor

__all__ = ["NSCF", "make_anchor", "train_anchor"]
__version__ = "0.1.0"
