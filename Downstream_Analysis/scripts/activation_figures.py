"""Drawing functions for Fig. 2 and Extended Data Figs. 2-4.

Every function draws one panel from the tables written by the extant_receptors notebooks,
writes it as SVG to the path it is given, and shows it inline or closes it.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Patch
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import gaussian_kde

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / "scripts"))

from plotting_functions import FIGSIZE, GREY  # noqa: E402
from activation_code import CLASS_SETS, CLUSTER_DESCRIPTORS, FUNCTIONAL_GROUPS, SHORT_THEME, significance_stars

CLASS_I_COLOUR = "#7f1734"
CLASS_II_COLOUR = "#557c99"
SHARED_COLOUR = "#9a8f80"
CLASS_SET_COLOURS = {"Class I only": CLASS_I_COLOUR, "shared": SHARED_COLOUR, "Class II only": CLASS_II_COLOUR}
CONFUSION_COLOURS = {"TP": "#355E3B", "FP": "#8A2F1D", "FN": "#5A3C2A", "TN": "#5F6A72"}
INK = "#6b6b67"
MUTED = "#8a8a86"
TEXT = "#4a4a46"
# Diverging ramp from the heatmap palette, so the effect matrices sit beside the heatmaps.
DIVERGING = LinearSegmentedColormap.from_list("div", ["#24877f", "#f2f0ea", "#8c4a1e"])

# Figure sizes in inches. Most figures use FIGSIZE. The panels of Fig. 2 have their own size, which gives
# their axes the proportions they have in that figure:
# the matrix of the heatmap is 2.04 times as wide as it is high, and the other panels are drawn at the
# size they have on the page, except the descriptor effects, which are an inset there.
HEATMAP_FIGSIZE = (FIGSIZE[0], 2.755)          # Fig. 2a
BREADTH_FIGSIZE = (4.026, 2.842)               # Fig. 2b
CLASS_GROUPS_FIGSIZE = (4.94, 4.573)           # Fig. 2c
CLASS_EFFECTS_FIGSIZE = (FIGSIZE[0], 2.915)    # Fig. 2c, inset
CLASS_TAGS_FIGSIZE = (2.854, 2.167)            # Fig. 2d
ENRICHMENT_FIGSIZE = (2.783, 4.325)            # Fig. 2e

# Text sizes in points, the same as in scripts/plotting_functions.py.
TITLE_SIZE = 8
SUBTITLE_SIZE = 7          # the second, statistics line under a title
LABEL_SIZE = 8
TICK_SIZE = 7
LEGEND_SIZE = 7
ANNOTATION_SIZE = 6.5


def save(fig, path, show):
    """Write the figure as an SVG at its figure size, then show it inline or close it."""
    fig.savefig(path)
    if show:
        plt.show()
    else:
        plt.close(fig)


def title_with_statistics(ax, title, statistics, title_colour, statistics_colour, statistics_size=SUBTITLE_SIZE,
                          fontweight="normal"):
    """The title on one line and the smaller statistics text under it, both centred on the axes."""
    n_lines = statistics.count("\n") + 1
    ax.set_title(title, fontsize=TITLE_SIZE, fontweight=fontweight, color=title_colour,
                 pad=statistics_size * 1.2 * n_lines + 7)
    ax.annotate(statistics, xy=(0.5, 1), xycoords="axes fraction", xytext=(0, 4), textcoords="offset points",
                ha="center", va="bottom", fontsize=statistics_size, color=statistics_colour)


# ---------------------------------------------------------------------------------------------
# Model check (Extended Data Fig. 2)
# ---------------------------------------------------------------------------------------------

def binding_rate_by_threshold(fine, predicted_rate, measured_rate, cut, path):
    """Predicted binding rate against threshold, with the measured rate and the rate-matched cut."""
    rate_colour = "#083d5e"
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(fine, predicted_rate * 100, color=rate_colour)
    ax.axhline(measured_rate * 100, ls="--", lw=1, color="#999999")
    ax.axvline(cut, ls=":", lw=1, color="#999999")
    ax.plot([cut], [measured_rate * 100], "o", color=rate_colour, ms=5)
    ax.annotate(f"{cut:.3f}", xy=(cut, measured_rate * 100), xytext=(6, 10),
                textcoords="offset points", fontsize=ANNOTATION_SIZE, color=rate_colour)
    ax.set_xlabel("Probability threshold", fontsize=LABEL_SIZE)
    ax.set_ylabel("Predicted binding proportion (%)", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title(f"Calibration: measured proportion {measured_rate:.2%}", fontsize=TITLE_SIZE)
    save(fig, path, True)


def metrics_by_threshold(coarse, mcc, f1, precision, recall, cut, path):
    """MCC, F1, precision and recall against threshold, with the rate-matched cut."""
    rate_colour, mcc_colour, f1_colour = "#083d5e", "#C8923B", "#4f585e"
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(coarse, mcc, color=mcc_colour, label="MCC")
    ax.plot(coarse, f1, color=f1_colour, label="F1")
    ax.plot(coarse, precision, color=rate_colour, lw=1, ls="--", label="Precision")
    ax.plot(coarse, recall, color=rate_colour, lw=1, ls=":", label="Recall")
    ax.axvline(cut, ls=":", lw=1, color="#999999")
    ax.set_xlabel("Probability threshold", fontsize=LABEL_SIZE)
    ax.set_ylabel("Score", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    ax.set_title("Discrimination agrees with calibration", fontsize=TITLE_SIZE)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    save(fig, path, True)


def binding_proportion_scatter(proportions, title, path=None):
    """Measured against predicted binding proportion, one point per receptor or odorant."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    pearson, _ = stats.pearsonr(proportions["exp_prop"], proportions["pred_prop"])
    spearman, _ = stats.spearmanr(proportions["exp_prop"], proportions["pred_prop"])
    ax.scatter(proportions["exp_prop"], proportions["pred_prop"], color="#000000", alpha=0.6, s=15,
               edgecolors="#999494", linewidths=0.3)
    slope, intercept = np.polyfit(proportions["exp_prop"], proportions["pred_prop"], 1)
    x_fit = np.linspace(0, 1, 200)
    ax.plot(x_fit, slope * x_fit + intercept, color="#C8923B", label=f"OLS  slope={slope:.2f}")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", alpha=0.6, label="y = x")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Experimental binding proportion", fontsize=LABEL_SIZE)
    ax.set_ylabel("Predicted binding proportion", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    title_with_statistics(ax, title, f"Pearson r = {pearson:.3f}   Spearman ρ = {spearman:.3f}",
                          title_colour=GREY, statistics_colour=MUTED)
    ax.legend(fontsize=LEGEND_SIZE, frameon=False)
    save(fig, path, True)


def confusion_matrix_figure(tn, fp, fn, tp, path):
    """The four confusion-matrix cells, each with its count and share of all pairs."""
    total = tn + fp + fn + tp
    values = np.array([[tn, fp], [fn, tp]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    colour_grid = np.array([[to_rgb(CONFUSION_COLOURS["TN"]), to_rgb(CONFUSION_COLOURS["FP"])],
                            [to_rgb(CONFUSION_COLOURS["FN"]), to_rgb(CONFUSION_COLOURS["TP"])]])

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.imshow(colour_grid, aspect="auto")
    ax.axvline(0.5, color="white", lw=1.6, zorder=5)
    ax.axhline(0.5, color="white", lw=1.6, zorder=5)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    for i in range(2):
        for j in range(2):
            count = values[i, j]
            ax.text(j, i, f"{labels[i, j]}\n{count:,}\n({count / total * 100:.1f}%)", ha="center",
                    va="center", fontsize=LABEL_SIZE, color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: 0", "Predicted: 1"], fontsize=TICK_SIZE)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True: 0", "True: 1"], fontsize=TICK_SIZE)
    save(fig, path, True)


def bounded_kde(values, bandwidth, grid):
    """Gaussian KDE on [0, 1] with a fixed bandwidth, reflected at both bounds.

    Reflecting keeps the mass that piles up near 0 inside the support, and a shared
    bandwidth makes the heights of different groups comparable.
    """
    values = np.asarray(values, dtype=float)
    mirrored = np.concatenate([values, -values, 2.0 - values])
    kde = gaussian_kde(mirrored, bw_method=bandwidth / mirrored.std(ddof=1))
    return kde(grid) * 3.0          # undo the threefold sample from mirroring


def probability_by_category(densities, counts, grid, figsize, path):
    """Density of the predicted probability within each confusion-matrix category."""
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    # Fills first, TP last so it sits on top; then full-opacity lines over them.
    for category in ["TN", "FN", "FP", "TP"]:
        ax.fill_between(grid, densities[category], color=CONFUSION_COLOURS[category], alpha=0.65,
                        linewidth=0)
    for category in ["TP", "FP", "FN", "TN"]:
        ax.plot(grid, densities[category], color=CONFUSION_COLOURS[category],
                label=f"{category}  (n={counts[category]:,})")
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted probability", fontsize=LABEL_SIZE)
    ax.set_ylabel("Density", fontsize=LABEL_SIZE)
    ax.tick_params(labelsize=TICK_SIZE)
    save(fig, path, True)


# ---------------------------------------------------------------------------------------------
# The activation matrix and its clusters (Fig. 2a,e, Extended Data Figs 3 and 4)
# ---------------------------------------------------------------------------------------------

def with_gaps(blocks, gap):
    """Join blocks of row or column indices into one list of slots, with `gap` empty slots (-1) between them."""
    slots = []
    for block in blocks:
        if len(block) == 0:
            continue
        if slots:
            slots += [-1] * gap
        slots += list(block)
    return np.array(slots)


def activation_heatmap(clustered_ids, clustered_matrix, labels, info, display_order, receptor_ids,
                       class_of, title, subtitle, path, row_order, show, tree_order=None, k=6):
    """The binary activation matrix, columns grouped by cluster.

    Rows are in co-tuning order (average-linkage on the Jaccard distance between receptors)
    or, with `row_order="tree"`, in the ladderised tip order of the tree. Within a cluster,
    columns are ordered by the same linkage; this only decides where a column is drawn.

    In co-tuning order the orphan receptors (no binding call) are the last rows, under a white
    gap. The white lines between clusters and above the orphans are empty rows and columns
    added to the image, so no receptor or odorant is drawn under them.
    """
    n_receptors, n_odorants = clustered_matrix.shape
    if row_order == "tree":
        row_of = {receptor: i for i, receptor in enumerate(receptor_ids)}
        row_blocks = [[row_of[receptor] for receptor in tree_order]]
    else:
        cotuning = leaves_list(linkage(squareform(np.nan_to_num(squareform(
            pdist(clustered_matrix, "jaccard")))), "average"))
        is_orphan = clustered_matrix.sum(1) == 0
        row_blocks = [[i for i in cotuning if not is_orphan[i]], [i for i in cotuning if is_orphan[i]]]

    column_blocks = []
    for c in display_order:
        members = np.where(labels == c)[0]
        if len(members) > 2:
            members = members[leaves_list(linkage(np.nan_to_num(
                pdist(clustered_matrix[:, members].T, "jaccard")), "average"))]
        column_blocks.append(members)

    # A gap is about 1.8 pt wide at this figure size.
    row_slots = with_gaps(row_blocks, gap=max(1, round(0.012 * n_receptors)))
    column_slots = with_gaps(column_blocks, gap=max(1, round(0.007 * n_odorants)))
    is_row = row_slots >= 0
    is_column = column_slots >= 0
    rows = row_slots[is_row]
    columns = column_slots[is_column]

    shown = np.full((len(row_slots), len(column_slots)), np.nan)
    shown[np.ix_(is_row, is_column)] = clustered_matrix[np.ix_(rows, columns)]
    white = np.ones(3)

    fig = plt.figure(figsize=HEATMAP_FIGSIZE, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0, hspace=0)
    grid = fig.add_gridspec(2, 3, width_ratios=[0.11, 4, 0.11], height_ratios=[0.14, 4])
    binding_colours = ListedColormap(["#F4F1EA", "#2E4057"])
    binding_colours.set_bad("white")
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(shown, aspect="auto", cmap=binding_colours, interpolation="nearest", rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel(f"{n_odorants} odorants — co-tuning cluster (Jaccard, PAM, k={k})", fontsize=LABEL_SIZE,
                  labelpad=4)

    cluster_strip = np.tile(white, (1, len(column_slots), 1))
    cluster_strip[0, is_column] = [to_rgb(info[labels[j]]["color"]) for j in columns]
    ax_top = fig.add_subplot(grid[0, 1], sharex=ax)
    ax_top.imshow(cluster_strip, aspect="auto", interpolation="nearest")
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    # Cluster names above the colour strip, one word per line so the narrow clusters do not collide.
    for c in display_order:
        positions = np.where(is_column)[0][labels[columns] == c]
        cluster_name = SHORT_THEME.get(info[c]["name"], info[c]["name"][:10])
        cluster_name = cluster_name.replace("·", "\n")
        ax_top.annotate(cluster_name, xy=(positions.mean(), 1), xycoords=("data", "axes fraction"),
                        xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                        fontsize=ANNOTATION_SIZE, linespacing=1.0, color=info[c]["color"], fontweight="bold",
                        annotation_clip=False)
    # The subtitle sits above the cluster names; the pad clears the two-line names.
    ax_top.set_title(subtitle, fontsize=SUBTITLE_SIZE, color=MUTED, pad=19)

    class_strip = np.tile(white, (len(row_slots), 1, 1))
    class_strip[is_row, 0] = [to_rgb(CLASS_I_COLOUR if class_of[receptor_ids[i]] == "I" else CLASS_II_COLOUR)
                              for i in rows]
    ax_class = fig.add_subplot(grid[1, 0], sharey=ax)
    ax_class.imshow(class_strip, aspect="auto", interpolation="nearest")
    ax_class.set_xticks([])
    ax_class.set_yticks([])
    for spine in ax_class.spines.values():
        spine.set_visible(False)
    # Side-strip labels aligned to the outer edge, so they run under the matrix, not off the figure.
    ax_class.set_xlabel("class", fontsize=ANNOTATION_SIZE, color=MUTED, labelpad=4, loc="left")

    breadth = np.full((len(row_slots), 1), np.nan)
    breadth[is_row, 0] = clustered_matrix.sum(1)[rows]
    breadth_colours = matplotlib.colormaps["Greys"].copy()
    breadth_colours.set_bad("white")
    ax_breadth = fig.add_subplot(grid[1, 2], sharey=ax)
    ax_breadth.imshow(breadth, aspect="auto", cmap=breadth_colours, interpolation="nearest", rasterized=True)
    ax_breadth.set_xticks([])
    ax_breadth.set_yticks([])
    for spine in ax_breadth.spines.values():
        spine.set_visible(False)
    ax_breadth.set_xlabel("breadth", fontsize=ANNOTATION_SIZE, color=MUTED, labelpad=4, loc="right")
    fig.legend(handles=[Patch(fc=CLASS_I_COLOUR, label="Class I"), Patch(fc=CLASS_II_COLOUR, label="Class II"),
                        Patch(fc="#2E4057", label="binding"), Patch(fc="#F4F1EA", ec="#bbb", label="no binding")],
               loc="outside lower center", ncol=4, frameon=False, fontsize=LEGEND_SIZE, handlelength=1.1,
               columnspacing=1.4)
    fig.suptitle(title, fontsize=TITLE_SIZE)
    save(fig, path, show)


def pcoa_figure(coordinates, variance, info, display_order, title, path, show):
    """The odorants on the first two principal coordinates of their Jaccard distances."""
    first = variance.pco1_percent.iloc[0]
    second = variance.pco2_percent.iloc[0]
    negative_share = variance.negative_percent.iloc[0]

    # Square axes with one scale on both coordinates, and room for the legend below: taller than FIGSIZE.
    fig, ax = plt.subplots(figsize=(FIGSIZE[0], 4.4), layout="constrained")
    ax.set_aspect("equal", adjustable="datalim")
    for c in display_order:
        points = coordinates[coordinates.cluster == c]
        ax.scatter(points.PCo1, points.PCo2, s=15, color=info[c]["color"], alpha=0.75, edgecolor="none",
                   label=f'{info[c]["name"]} (n={info[c]["size"]})', zorder=3)
    ax.axhline(0, color="#e5e5e2", lw=0.6, zorder=0)
    ax.axvline(0, color="#e5e5e2", lw=0.6, zorder=0)
    ax.set_xlabel(f"PCo1 ({first:.1f}%)", fontsize=LABEL_SIZE)
    ax.set_ylabel(f"PCo2 ({second:.1f}%)", fontsize=LABEL_SIZE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=TICK_SIZE)
    # Legend below the axes, outside the points.
    fig.legend(frameon=False, fontsize=ANNOTATION_SIZE, loc="outside lower center", ncol=2, 
               handletextpad=0.2, columnspacing=1.2, borderaxespad=0.2)
    # The eigenvalue note sits under the title, off the points.
    title_with_statistics(ax, title, f"non-Euclidean (neg. eigenvalue mass {negative_share:.0f}%)",
                          title_colour=GREY, statistics_colour=MUTED, statistics_size=ANNOTATION_SIZE)
    save(fig, path, show)


def enrichment_bars(significant, info, display_order, title, path, show):
    """Fold enrichment of every significant descriptor, grouped and coloured by cluster."""
    rows = []
    for c in display_order:
        for row in significant[significant.cluster == c].sort_values("obs_exp", ascending=False).itertuples():
            rows.append((c, row.tag, row.obs_exp, row.fdr))
    if not rows:
        return
    ys, tag_labels, colours, folds, q_values = [], [], [], [], []
    y = 0
    previous = None
    cluster_gap = 1.0          # extra space between clusters, in rows
    for c, tag, fold, q in rows:
        if previous is not None and c != previous:
            y -= cluster_gap
        ys.append(y)
        tag_labels.append(tag)
        colours.append(info[c]["color"])
        folds.append(fold)
        q_values.append(q)
        y -= 1
        previous = c
    ys = np.array(ys)
    folds = np.array(folds)

    fig, ax = plt.subplots(figsize=ENRICHMENT_FIGSIZE, layout="constrained")
    ax.barh(ys, folds, color=colours, alpha=0.92, height=0.74)
    ax.axvline(1, color=MUTED, ls="--", lw=1.0)
    x_max = folds.max()
    # Nudged down: asterisks sit high in the glyph box, so 'center' reads as floating.
    for y_value, fold, q in zip(ys, folds, q_values):
        ax.text(fold + x_max * 0.02, y_value - 0.12, significance_stars(q), va="center", ha="left",
                fontsize=ANNOTATION_SIZE, color=MUTED)
    # About 30 descriptors share the height of the panel, so their names are smaller than other tick labels,
    # and smaller again when there are more.
    ax.set_yticks(ys)
    ax.set_yticklabels(tag_labels, fontsize=min(5.5, 165 / len(rows)))
    for tick_label, colour in zip(ax.get_yticklabels(), colours):
        tick_label.set_color(colour)
    ax.set_xlim(0, x_max * 1.14)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlabel("fold enrichment of odour descriptor\n(observed / expected)", fontsize=LABEL_SIZE)
    # Title on one line, the test and its significance key on a second, smaller line.
    title_with_statistics(ax, title, "hypergeometric test, BH\n* < 0.05    ** < 0.01    *** < 0.001",
                          title_colour=GREY, statistics_colour=MUTED)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False, pad=1.5)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    save(fig, path, show)


def _cluster_tags(info, display_order):
    colours = [info[c]["color"] for c in display_order]
    tags = ["C%d" % (i + 1) for i in range(len(display_order))]
    names = [SHORT_THEME.get(info[c]["name"], info[c]["name"]) for c in display_order]
    return colours, tags, names


def cluster_functional_groups(tests, info, display_order, name, path, show):
    """Share of each cluster carrying each group, with the stars of `cluster_group_tests`."""
    groups = list(dict.fromkeys(tests.group))
    colours, tags, names = _cluster_tags(info, display_order)
    grid = tests.set_index(["group", "cluster"])
    percent = np.array([[grid.loc[(g, c), "percent"] for c in display_order] for g in groups])
    q_values = np.array([[grid.loc[(g, c), "q"] for c in display_order] for g in groups])
    order = np.argsort(percent.mean(1))          # rarest group at the bottom
    groups = [groups[i] for i in order]
    percent = percent[order]
    q_values = q_values[order]

    ys = np.arange(len(groups))
    bar_height = 0.84 / len(display_order)
    x_max = percent.max()
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for j, c in enumerate(display_order):
        offset = ((len(display_order) - 1) / 2 - j) * bar_height     # C1 at the top of each group
        ax.barh(ys + offset, percent[:, j], height=bar_height * 0.9, color=colours[j], alpha=0.92,
                label="%s  %s" % (tags[j], names[j]))
        for i in range(len(groups)):
            stars = significance_stars(q_values[i, j])
            if stars:
                # Six bars share each row, so the stars are a size smaller than other annotations.
                ax.text(percent[i, j] + x_max * 0.012, ys[i] + offset - 0.06, stars, va="center", ha="left",
                        fontsize=5.5, color=MUTED)
    for i in ys[:-1]:
        ax.axhline(i + 0.5, color="#ecebe6", lw=0.8, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(groups, fontsize=TICK_SIZE)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.set_xlim(0, x_max * 1.10)
    ax.set_xlabel("% of cluster carrying the group", fontsize=LABEL_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    # Title on one line, the test and its significance key on a second, smaller line.
    title_with_statistics(ax, "Functional groups — %s" % name,
                          "Fisher one-vs-rest, BH:   * < 0.05    ** < 0.01    *** < 0.001",
                          title_colour=GREY, statistics_colour=MUTED)
    # Legend below the axes, outside the bars.
    fig.legend(frameon=False, fontsize=ANNOTATION_SIZE, loc="outside lower center", ncol=3, handlelength=1.2,
               handletextpad=0.4, columnspacing=1.0)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    save(fig, path, show)


def cluster_effects(effects, info, display_order, name, path, show):
    """Cliff's delta of each cluster against all other odorants, with the stars of `cluster_descriptor_effects`."""
    keys = list(dict.fromkeys(effects.descriptor))
    labels = list(dict.fromkeys(effects.label))
    colours, tags, names = _cluster_tags(info, display_order)
    grid = effects.set_index(["descriptor", "cluster"])
    delta = np.array([[grid.loc[(k, c), "cliffs_delta"] for c in display_order] for k in keys])
    q_values = np.array([[grid.loc[(k, c), "q"] for c in display_order] for k in keys])
    limit = np.abs(delta).max()

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    image = ax.imshow(delta, cmap=DIVERGING, aspect="auto", vmin=-limit, vmax=limit)
    # Each cell shows its delta, with the stars on a second line underneath so the cells stay readable.
    for i in range(len(keys)):
        for j in range(len(display_order)):
            stars = significance_stars(q_values[i, j])
            cell_text = "%+.2f" % delta[i, j]
            if stars:
                cell_text = cell_text + "\n" + stars
            ax.text(j, i, cell_text, ha="center", va="center", fontsize=ANNOTATION_SIZE, linespacing=0.85,
                    color="white" if abs(delta[i, j]) > 0.62 * limit else TEXT)
    # Cluster names one word per line, so neighbouring tick labels do not run into each other.
    tick_names = []
    for t, n in zip(tags, names):
        tick_names.append(t + "\n" + n.replace("·", "\n"))
    ax.set_xticks(range(len(display_order)))
    ax.set_xticklabels(tick_names, fontsize=6.5, linespacing=1.0)
    for tick_label, colour in zip(ax.get_xticklabels(), colours):
        tick_label.set_color(colour)
    # Descriptor names wrap before their parenthesis, which leaves the cells wide enough for the cluster names.
    wrapped_labels = []
    for label in labels:
        wrapped_labels.append(label.replace(" (", "\n("))
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(wrapped_labels, fontsize=TICK_SIZE, linespacing=0.95)
    ax.tick_params(axis="both", which="major", pad=2)
    ax.set_xticks(np.arange(-.5, len(display_order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colourbar.set_label("Cliff's delta, cluster vs all other molecules", fontsize=ANNOTATION_SIZE)
    colourbar.ax.tick_params(labelsize=TICK_SIZE)
    colourbar.outline.set_visible(False)
    # Title on one line, the test and its significance key on a second, smaller line.
    title_with_statistics(ax, "What each cluster is made of — %s" % name,
                          "Mann-Whitney one-vs-rest, BH:   * < 0.05    ** < 0.01    *** < 0.001",
                          title_colour=GREY, statistics_colour=MUTED)
    save(fig, path, show)


def cluster_summary(descriptors, info, display_order):
    """Per cluster: size, median of each descriptor, % carrying each functional group."""
    keys = [key for key, _ in CLUSTER_DESCRIPTORS]
    _, tags, _ = _cluster_tags(info, display_order)
    rows = []
    for j, c in enumerate(display_order):
        members = descriptors[descriptors.cluster == c]
        row = dict(cluster=tags[j], theme=info[c]["name"], n=len(members))
        row.update({key: round(float(np.median(members[key])), 3) for key in keys})
        row.update({group: round(100 * float(members[group].mean()), 1) for group in FUNCTIONAL_GROUPS})
        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------------------
# Tuning breadth by class (Fig. 2b)
# ---------------------------------------------------------------------------------------------

BREADTH_EDGES = np.array([0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, np.inf])
BREADTH_LABELS = ["0", "1", "2-3", "4-7", "8-15", "16-31", "32-63", "64-127", "128-255", "256-511", "512+"]
CLASS_COLOURS = {"Class I": CLASS_I_COLOUR, "Class II": CLASS_II_COLOUR}


def _breadth_title(test):
    row = test[test.comparison == "all receptors"].iloc[0]
    return (f"Mann-Whitney $P$ = {row.mann_whitney_p:.3g} {significance_stars(row.mann_whitney_p)},  "
            f"Cliff's $\\delta$ = {row.cliffs_delta:+.2f},  shift {row.shift_ligands:+.0f} ligands "
            f"[95% CI {row.shift_low:+.0f}, {row.shift_high:+.0f}]")


def breadth_histogram(breadth, test, path, show=True):
    """Receptors per ligand-count bin, as % of each class. Bin width doubles; 0 has its own bar.

    Percent rather than counts, because on a shared count axis the 62 Class I receptors are
    a sliver next to 371 Class II. The counts are printed above the bars.
    """
    series = {k: breadth.loc[breadth.OR_class == k, "n_ligands"].values for k in ["Class I", "Class II"]}
    counts = {k: np.bincount(np.searchsorted(BREADTH_EDGES, v, side="right") - 1,
                             minlength=len(BREADTH_LABELS)) for k, v in series.items()}
    last = max(np.nonzero(c)[0].max() for c in counts.values()) + 1
    counts = {k: c[:last] for k, c in counts.items()}
    x = np.arange(last)
    percent = {k: 100 * c / len(series[k]) for k, c in counts.items()}
    legend_text = {k: f"{k}    n={len(v)}, median {np.median(v):g}, {(v == 0).sum()} orphans"
                   for k, v in series.items()}

    width = 0.40
    fig, ax = plt.subplots(figsize=BREADTH_FIGSIZE, layout="constrained")
    for offset, k in [(-width / 2, "Class I"), (width / 2, "Class II")]:
        ax.bar(x + offset, percent[k], width=width, color=CLASS_COLOURS[k], alpha=0.95, zorder=3,
               label=legend_text[k])
        for xi, (pct, n) in enumerate(zip(percent[k], counts[k])):
            if n:
                ax.text(xi + offset, pct + 0.4, str(n), ha="center", va="bottom", fontsize=ANNOTATION_SIZE,
                        color=CLASS_COLOURS[k], zorder=4)
    ax.set_ylim(0, max(max(p) for p in percent.values()) * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels(BREADTH_LABELS[:last], fontsize=TICK_SIZE, rotation=40, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.7, last - 0.3)
    ax.set_xlabel("ligands per receptor   (bin width doubles; 0 = orphan receptors)", fontsize=LABEL_SIZE)
    ax.set_ylabel("% of class", fontsize=LABEL_SIZE)
    ax.tick_params(axis="y", labelsize=TICK_SIZE)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(legend_text[k]) for k in ["Class I", "Class II"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=ANNOTATION_SIZE,
              loc="upper right", handlelength=1.2, borderaxespad=0.2)
    # Title on one line, the test statistics on a second, smaller line.
    title_with_statistics(ax, "Ligands per receptor by class", _breadth_title(test), title_colour=GREY,
                          statistics_colour=MUTED, statistics_size=ANNOTATION_SIZE)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.supxlabel("bar height = % of that class;  number above each bar = receptors", fontsize=ANNOTATION_SIZE,
                  color=MUTED)
    save(fig, path, show)


# ---------------------------------------------------------------------------------------------
# What each class reads (Fig. 2c,d)
# ---------------------------------------------------------------------------------------------

def class_functional_groups(tests, name, path, show):
    """Share of each class set carrying each functional group, with the stars of `class_group_tests`."""
    tests = tests.sort_values("Class I only", kind="stable").reset_index(drop=True)
    ys = np.arange(len(tests))
    bar_height = 0.26
    fig, ax = plt.subplots(figsize=CLASS_GROUPS_FIGSIZE, layout="constrained")
    for offset, g in zip([bar_height, 0, -bar_height], CLASS_SETS):
        ax.barh(ys + offset, tests[g], height=bar_height, color=CLASS_SET_COLOURS[g], alpha=0.9, label=g)
    x_max = tests[CLASS_SETS].max().max()
    for y, q in zip(ys, tests.q):
        ax.text(x_max * 1.02, y, significance_stars(q) or "ns", va="center", fontsize=ANNOTATION_SIZE, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels(tests.group, fontsize=TICK_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.set_xlabel("% of odorants containing group", fontsize=LABEL_SIZE)
    ax.set_xlim(0, x_max * 1.15)
    # Legend below the x label, outside the bars.
    fig.legend(frameon=False, fontsize=LEGEND_SIZE, loc="outside lower center", ncol=3, handlelength=1.2)
    # Title on one line, what the stars test on a second, smaller line.
    title_with_statistics(ax, "Functional groups — %s" % name, "significance: I-only vs II-only, BH",
                          title_colour=GREY, statistics_colour=MUTED)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    save(fig, path, show)


def cliffs_figure(effects, title, path, show):
    """Cliff's delta per descriptor, sorted by effect, with the stars of `class_descriptor_effects`."""
    effects = effects.sort_values("cliffs_delta", kind="stable").reset_index(drop=True)
    values = effects.cliffs_delta.values
    fig, ax = plt.subplots(figsize=CLASS_EFFECTS_FIGSIZE, layout="constrained")
    ax.barh(range(len(effects)), values, color=[CLASS_I_COLOUR if e > 0 else CLASS_II_COLOUR for e in values],
            alpha=0.9)
    ax.axvline(0, color="#c9c9c4", lw=0.8)
    x_range = max(abs(values.min()), abs(values.max()), 0.05)
    for i, (effect, q) in enumerate(zip(values, effects.q)):
        ax.text(effect + 0.03 * x_range * (1 if effect >= 0 else -1), i, significance_stars(q) or "ns",
                va="center", ha="left" if effect >= 0 else "right", fontsize=ANNOTATION_SIZE, color=MUTED)
    ax.set_yticks(range(len(effects)))
    ax.set_yticklabels(effects.label, fontsize=TICK_SIZE)
    ax.tick_params(axis="x", labelsize=TICK_SIZE)
    ax.set_xlim(-x_range * 1.3, x_range * 1.3)
    ax.set_xlabel("effect size (Cliff delta)\n$\\leftarrow$ Class II only higher   |   "
                  "Class I only higher $\\rightarrow$", fontsize=LABEL_SIZE)
    ax.set_title(title, fontsize=TITLE_SIZE)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    save(fig, path, show)


def class_tag_bars(tags, path, show):
    """The odour descriptors that differ between the two exclusive class sets.

    Class I only up, class II only down, as % of each set's tagged odorants; the number of
    odorants at each bar tip and the Benjamini-Hochberg stars of `tag_percentages` above.

    Parameters
    ----------
    tags : DataFrame
        The exclusive-set rows of `class_odour_tags.csv`.
    """
    shown = tags[tags.q < 0.05].sort_values("gap", ascending=False, kind="stable").reset_index(drop=True)
    n_tagged_I, n_tagged_II = int(tags.n_tagged_I.iloc[0]), int(tags.n_tagged_II.iloc[0])
    x = np.arange(len(shown))
    fig, ax = plt.subplots(figsize=CLASS_TAGS_FIGSIZE, layout="constrained")
    ax.bar(x, shown.pct_I, width=0.62, color=CLASS_I_COLOUR, alpha=0.92, zorder=3)
    ax.bar(x, -shown.pct_II, width=0.62, color=CLASS_II_COLOUR, alpha=0.92, zorder=3)
    ax.axhline(0, color="#9a9a96", lw=0.8, zorder=4)

    top, bottom = shown.pct_I.max() * 1.80, shown.pct_II.max() * 1.30
    for xi, row in zip(x, shown.itertuples()):
        ax.text(xi, row.pct_I + top * 0.025, "%d" % row.n_I, ha="center", va="bottom", fontsize=LEGEND_SIZE,
                color=CLASS_I_COLOUR)
        ax.text(xi, -row.pct_II - bottom * 0.025, "%d" % row.n_II, ha="center", va="top", fontsize=LEGEND_SIZE,
                color=CLASS_II_COLOUR)
        ax.text(xi, row.pct_I + top * 0.185, significance_stars(row.q), ha="center", va="bottom",
                fontsize=LEGEND_SIZE, color=MUTED)
    ax.set_ylim(-bottom, top)
    ax.set_xlim(-0.66, len(shown) - 0.34)
    ticks = [v for v in range(-50, 51, 10) if -bottom <= v <= top]
    ax.set_yticks(ticks)
    ax.set_yticklabels(["%d%%" % abs(v) for v in ticks], fontsize=TICK_SIZE)
    ax.set_ylabel("% of odorants", fontsize=LABEL_SIZE, labelpad=2)
    ax.set_xticks(x)
    ax.set_xticklabels(shown.tag, fontsize=TICK_SIZE, rotation=30, ha="right", rotation_mode="anchor")
    ax.tick_params(axis="x", length=0, pad=1.0)
    for side in ["top", "right", "bottom"]:
        ax.spines[side].set_visible(False)
    ax.text(len(shown) - 0.34, top * 0.99, "class I only  n=%d" % n_tagged_I, ha="right", va="top",
            fontsize=LEGEND_SIZE, color=CLASS_I_COLOUR)
    ax.text(len(shown) - 0.34, top * 0.82, "class II only  n=%d" % n_tagged_II, ha="right", va="top",
            fontsize=LEGEND_SIZE, color=CLASS_II_COLOUR)
    save(fig, path, show)


