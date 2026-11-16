"""Plot treatment merges recorded on a fusion-penalty grid."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np


@dataclass(frozen=True)
class _Branch:
    leaves: tuple
    penalty: float = 0.0
    children: tuple = ()


def _fusion_forest(penalties, partitions, *, truncate_at_split=False):
    active = [_Branch((i,)) for i in range(partitions.shape[1])]
    for index, (penalty, labels) in enumerate(zip(penalties, partitions)):
        groups = {}
        for branch in active:
            group = labels[branch.leaves[0]]
            if not np.all(labels[list(branch.leaves)] == group):
                if truncate_at_split:
                    return active, index
                raise ValueError(
                    f"Fusion path is not nested at lambda={penalty:g}; "
                    "inspect the recorded partitions to see the split."
                )
            groups.setdefault(group, []).append(branch)
        active = [
            children[0] if len(children) == 1 else _Branch(
                tuple(leaf for child in children for leaf in child.leaves),
                float(penalty), tuple(children),
            )
            for children in groups.values()
        ]
    return active, len(penalties)


def _height(penalty):
    if penalty == 0:
        return 0.0
    return float(np.logaddexp(0, np.log(penalty) + np.log(1000)))


def plot_fusion_dendrogram(penalties, partitions, treatments, *, selected_index=None,
                           true_groups=None, truncate_at_split=True, ax=None):
    """Draw a dendrogram from recorded treatment partitions.

    Parameters
    ----------
    penalties : array-like of shape (n_penalties,)
        Evaluated fusion penalties in nondecreasing order.
    partitions : array-like of shape (n_penalties, n_treatments)
        Group labels in treatment order at each penalty. Equal labels in a
        row indicate membership in the same group.
    treatments : array-like of shape (n_treatments,)
        Labels to show at the leaves.
    selected_index : int or None, default=None
        Index of the evaluated penalty to mark with a display cut if it
        lies in the plotted path.
    true_groups : array-like of shape (n_treatments,), optional
        Reference group labels for coloring branches, in treatment order.
    truncate_at_split : bool, default=True
        Show the nested prefix and annotate the first split. Set to False
        to raise a ValueError when a group splits.
    ax : matplotlib.axes.Axes, optional
        Target axes. Create new axes when omitted.

    Returns
    -------
    matplotlib.axes.Axes
        Dendrogram with merge heights log(1 + 1000 * lambda). Dots mark
        mergers of three or more branches at the same grid point.
    """
    penalties = np.asarray(penalties, dtype=float)
    partitions = np.asarray(partitions)
    if (penalties.ndim != 1 or not len(penalties) or not np.isfinite(penalties).all()
            or np.any(penalties < 0) or np.any(penalties[1:] < penalties[:-1])):
        raise ValueError("Fusion penalties must be finite, nonnegative and in increasing order")
    if not len(treatments) or partitions.shape != (len(penalties), len(treatments)):
        raise ValueError("Provide one partition aligned with treatments for each fusion penalty")
    if true_groups is not None and np.shape(true_groups) != (len(treatments),):
        raise ValueError("true_groups must contain one label per treatment")
    if selected_index is not None and (
        isinstance(selected_index, (bool, np.bool_)) or not isinstance(selected_index, Integral)
        or not 0 <= selected_index < len(penalties)
    ):
        raise ValueError("selected_index must be a valid index into penalties")

    forest, stop = _fusion_forest(penalties, partitions, truncate_at_split=truncate_at_split)
    split_height = None
    xlabel = "Treatment"
    if stop < len(penalties):
        split_height = _height(penalties[stop])
        xlabel += f"\nPath truncated: groups split at lambda = {penalties[stop]:.6g}."
        if selected_index is not None and selected_index >= stop:
            xlabel += f"\nSelected lambda = {penalties[selected_index]:.6g} lies beyond the plotted prefix."
            selected_index = None
        penalties, partitions = penalties[:stop], partitions[:stop]
    selected_lambda = penalties[selected_index] if selected_index is not None else None

    from matplotlib import pyplot as plt

    if ax is None:
        _, ax = plt.subplots(layout="constrained")
    default_color = "#3e536d"
    if true_groups is None:
        leaf_colors = [default_color] * len(treatments)
    else:
        palette = plt.get_cmap("tab10").colors
        colors = {group: palette[i % len(palette)] for i, group in enumerate(dict.fromkeys(true_groups))}
        leaf_colors = [colors[group] for group in true_groups]
    order = [leaf for root in forest for leaf in root.leaves]
    positions = {leaf: i for i, leaf in enumerate(order)}
    ceiling = _height(penalties[-1] if len(forest) > 1 else forest[0].penalty)
    if selected_lambda is not None:
        ceiling = max(ceiling, _height(selected_lambda))
    padding = max(ceiling, 1) * 0.06
    if selected_lambda is not None:
        height = _height(selected_lambda)
        n_selected = len(set(partitions[selected_index]))
        offset = padding / 2
        if split_height is not None:
            offset = min(offset, 0.4 * (split_height - height))
        for penalty, labels in zip(penalties[selected_index + 1:], partitions[selected_index + 1:]):
            if len(set(labels)) < n_selected:
                offset = min(offset, 0.4 * (_height(penalty) - height))
                break
        cut = height + offset
        if len(forest) > 1:
            ceiling = max(ceiling, cut + offset / 2)
        label = f"Selected lambda = {selected_lambda:.6g}"
        if offset > 0:
            label = f"Display cut (selected lambda = {selected_lambda:.6g})"

    def draw(branch):
        color = leaf_colors[branch.leaves[0]]
        if any(leaf_colors[leaf] != color for leaf in branch.leaves):
            color = default_color
        if not branch.children:
            return positions[branch.leaves[0]], 0.0, color
        children = [draw(child) for child in branch.children]
        height = _height(branch.penalty)
        for x, start, child_color in children:
            ax.plot([x, x], [start, height], color=child_color, linewidth=1.5)
        left, right = children[0][0], children[-1][0]
        ax.plot([left, right], [height, height], color=color, linewidth=1.5)
        center = (left + right) / 2
        if len(children) > 2:
            ax.plot(center, height, marker="o", markersize=4.5, color=color,
                    markeredgecolor="white", markeredgewidth=0.6, zorder=4)
        return center, height, color

    for root in forest:
        x, height, color = draw(root)
        if height < ceiling:
            ax.plot([x, x], [height, ceiling], color=color, linewidth=1.5)

    if selected_lambda is not None:
        ax.axhline(cut, color="#bd3f3e", linestyle=":", linewidth=1.5, label=label)
        ax.legend(loc="upper left", frameon=False)
    ax.set_xticks(range(len(order)), [treatments[i] for i in order])
    for label, leaf in zip(ax.get_xticklabels(), order):
        label.set_color(leaf_colors[leaf])
    ax.set(xlim=(-0.55, len(order) - 0.45), ylim=(-padding, ceiling + 2 * padding),
           xlabel=xlabel, ylabel=r"Fusion height $\log(1 + 1000\lambda)$")
    ax.spines[["top", "right"]].set_visible(False)
    return ax
