"""Private Python treatment-policy interfaces."""

from importlib.util import find_spec as _find_spec

__version__ = "0.2.37"
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
                wheel = f"itr_methods_private-{__version__}-py3-none-any.whl"
                raise ImportError(
                    f'NSCF requires PyTorch. From the wheel directory run '
                    f'python -m pip install "./{wheel}[nscf]".'
                ) from exc
            raise
        return NSCF
    raise AttributeError(name)
