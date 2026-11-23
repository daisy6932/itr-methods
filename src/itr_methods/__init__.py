"""Python treatment-policy interfaces for SCAF and NSCF."""

from importlib.util import find_spec as _find_spec

__version__ = "0.2.38"
__all__ = ["SCAF"]
if _find_spec("torch") is not None:
    __all__.append("NSCF")


def __dir__():
    return sorted(set(globals()) | {"SCAF", "NSCF"})


def __getattr__(name):
    if name == "SCAF":
        from .scaf import SCAF

        return SCAF
    if name == "NSCF":
        try:
            from ._nscf import NSCF
        except ModuleNotFoundError as exc:
            if exc.name == "torch":
                raise ImportError(
                    'NSCF requires PyTorch. Install it in this Python environment '
                    'with: python -m pip install "torch>=2.7".'
                ) from exc
            raise
        return NSCF
    raise AttributeError(name)
