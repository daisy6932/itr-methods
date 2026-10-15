# Attribution and release scope

This private repository packages the project owner's NSCF implementation. No
blanket open-source license or exclusive copyright declaration is added.

## External software

PyTorch, NumPy, SciPy, Matplotlib, setuptools, and wheel are installed separately
and retain their own licenses. Their implementations are not copied into this
repository. Removing a comparator does not remove dependency-license duties.

The coordinatewise L1 proximal calculation uses SciPy's isotonic regression.
The source preserves its mathematical reference to Lin et al.,
arXiv:1808.07181, Propositions 1–2. Mathematical attribution is distinct from
permission to copy another implementation.

## Excluded comparator

SCAF is a related comparator, not the implementation distributed here:
Ma, H., Zeng, D., and Liu, Y. (2022), *Learning Individualized Treatment Rules
with Many Treatments: A Supervised Clustering Approach Using Adaptive Fusion*,
NeurIPS 35, 15956–15969.

No SCAF R/C++ source or worker, benchmark runner, comparison dataset, or original
Git history is included. The paper's comparisons remain part of the research
record; this repository is not an implementation of those external methods.

## Data and licensing

The example uses generated synthetic data. No PDX or other raw research dataset
is redistributed. The paper's Creative Commons license is not automatically a
software or dataset license. Public visibility and an open-source license must
be decided separately after applicable permissions are confirmed.
