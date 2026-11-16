"""Report installed dependencies for SCAF and the optional NSCF estimator."""

import argparse
from importlib import metadata
import json
import platform


def inspect():
    """Return interpreter and dependency availability as a dictionary."""
    packages = {}
    for name in ("numpy", "scipy", "scikit-learn", "torch", "matplotlib"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    return {
        "package": "itr-methods-private",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "python_packages": packages,
        "scaf_available": all(packages[p] for p in ("numpy", "scipy", "scikit-learn")),
        "nscf_available": all(packages[p] for p in ("numpy", "scipy", "scikit-learn", "torch")),
    }


def main():
    """Print dependency metadata and set a status for the requested estimator."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require", choices=["all", "scaf", "nscf"], default="all")
    args = parser.parse_args()
    report = inspect()
    print(json.dumps(report, indent=2))
    required = ["scaf", "nscf"] if args.require == "all" else [args.require]
    raise SystemExit(0 if all(report[name + "_available"] for name in required) else 1)
