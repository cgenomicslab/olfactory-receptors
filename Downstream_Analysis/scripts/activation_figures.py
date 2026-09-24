"""Figures for the human receptor activation code (Figure 2 and its supplementary notebooks).

Every function draws one figure, writes it as SVG to the path it is given, and either shows
it inline (`show=True`) or closes it. The numbers come from `activation_code.py`.
"""

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, to_rgb
from matplotlib.patches import Patch, Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import pdist, squareform
from scipy.stats import fisher_exact, gaussian_kde, kruskal, mannwhitneyu

from activation_code import (CLASS_SETS, CLUSTER_DESCRIPTORS, FUNCTIONAL_GROUPS, SHORT_THEME,
                             benjamini_hochberg, cliffs_delta, jaccard_distance_between_columns,
                             pcoa, significance_stars)

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


def save(fig, path, show):
    """Write the figure as SVG, then show it inline or close it."""
    fig.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------------------------
# Model check (Extended Data Fig. 2)
# ---------------------------------------------------------------------------------------------

def threshold_calibration(fine, predicted_rate, coarse, mcc, f1, precision, recall, measured_rate,
                          cut, figsize, path):
    """Left: predicted binding rate against threshold. Right: MCC, F1, precision, recall."""
    rate_colour, mcc_colour, f1_colour = "#083d5e", "#C8923B", "#4f585e"
    fig, axes = plt.subplots(1, 2, figsize=(2 * figsize[0], figsize[1]))

    ax = axes[0]
    ax.plot(fine, predicted_rate * 100, color=rate_colour, lw=2)
    ax.axhline(measured_rate * 100, ls="--", lw=1, color="#999999")
    ax.axvline(cut, ls=":", lw=1, color="#999999")
    ax.plot([cut], [measured_rate * 100], "o", color=rate_colour, ms=5)
    ax.annotate(f"{cut:.3f}", xy=(cut, measured_rate * 100), xytext=(6, 10),
                textcoords="offset points", fontsize=8, color=rate_colour)
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Predicted binding proportion (%)")
    ax.set_title(f"Calibration: measured proportion {measured_rate:.2%}", fontsize=9)

    ax = axes[1]
    ax.plot(coarse, mcc, color=mcc_colour, lw=2, label="MCC")
    ax.plot(coarse, f1, color=f1_colour, lw=2, label="F1")
    ax.plot(coarse, precision, color=rate_colour, lw=1, ls="--", label="Precision")
    ax.plot(coarse, recall, color=rate_colour, lw=1, ls=":", label="Recall")
    ax.axvline(cut, ls=":", lw=1, color="#999999")
    ax.set_xlabel("Probability threshold")
    ax.set_ylabel("Score")
    ax.set_title("Discrimination agrees with calibration", fontsize=9)
    ax.legend(fontsize=7, frameon=False)

    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.show()


def binding_proportion_scatter(proportions, title, path=None):
    """Measured against predicted binding proportion, one point per receptor or odorant."""
    from scipy import stats

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    pearson, _ = stats.pearsonr(proportions["exp_prop"], proportions["pred_prop"])
    spearman, _ = stats.spearmanr(proportions["exp_prop"], proportions["pred_prop"])
    ax.scatter(proportions["exp_prop"], proportions["pred_prop"], color="#000000", alpha=0.6, s=15,
               edgecolors="#999494", linewidths=0.3)
    slope, intercept = np.polyfit(proportions["exp_prop"], proportions["pred_prop"], 1)
    x_fit = np.linspace(0, 1, 200)
    ax.plot(x_fit, slope * x_fit + intercept, color="#C8923B", lw=2, label=f"OLS  slope={slope:.2f}")
    ax.plot([0, 1], [0, 1], color="#999999", lw=1, ls="--", alpha=0.6, label="y = x")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Experimental binding proportion")
    ax.set_ylabel("Predicted binding proportion")
    ax.set_title(f"{title}\nPearson r = {pearson:.3f}   Spearman ρ = {spearman:.3f}")
    ax.legend(fontsize=9)
    # Saved before tight_layout, as the published panel was.
    if path is not None:
        plt.savefig(path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


def confusion_matrix_figure(tn, fp, fn, tp, path):
    """The four confusion-matrix cells, each with its count and share of all pairs."""
    total = tn + fp + fn + tp
    values = np.array([[tn, fp], [fn, tp]])
    labels = np.array([["TN", "FP"], ["FN", "TP"]])
    colour_grid = np.array([[to_rgb(CONFUSION_COLOURS["TN"]), to_rgb(CONFUSION_COLOURS["FP"])],
                            [to_rgb(CONFUSION_COLOURS["FN"]), to_rgb(CONFUSION_COLOURS["TP"])]])

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.imshow(colour_grid, aspect="auto")
    ax.axvline(0.5, color="black", lw=2, zorder=5)
    ax.axhline(0.5, color="black", lw=2, zorder=5)
    for spine in ax.spines.values():
        spine.set_linewidth(2)
    for i in range(2):
        for j in range(2):
            count = values[i, j]
            ax.text(j, i, f"{labels[i, j]}\n{count:,}\n({count / total * 100:.1f}%)", ha="center",
                    va="center", fontsize=14, fontweight="bold", color="white")
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Predicted: 0", "Predicted: 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True: 0", "True: 1"])
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


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
    fig, ax = plt.subplots(figsize=figsize)
    # Fills first, TP last so it sits on top; then full-opacity lines over them.
    for category in ["TN", "FN", "FP", "TP"]:
        ax.fill_between(grid, densities[category], color=CONFUSION_COLOURS[category], alpha=0.65,
                        linewidth=0)
    for category in ["TP", "FP", "FN", "TN"]:
        ax.plot(grid, densities[category], color=CONFUSION_COLOURS[category],
                label=f"{category}  (n={counts[category]:,})", linewidth=2.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Density")
    plt.savefig(path, bbox_inches="tight")
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------------------------
# The activation matrix and its clusters (Fig. 2a,e, Extended Data Figs 3 and 4)
# ---------------------------------------------------------------------------------------------

def activation_heatmap(clustered_ids, clustered_matrix, labels, info, display_order, receptor_ids,
                       class_of, title, subtitle, path, row_order, show, tree_order=None, k=6):
    """The binary activation matrix, columns grouped by cluster.

    Rows are in co-tuning order (average-linkage on the Jaccard distance between receptors)
    or, with `row_order="tree"`, in the ladderised tip order of the tree. Within a cluster,
    columns are ordered by the same linkage; this only decides where a column is drawn.
    """
    if row_order == "tree":
        row_of = {receptor: i for i, receptor in enumerate(receptor_ids)}
        row_positions = np.array([row_of[receptor] for receptor in tree_order])
        ylabel = "433 human ORs (phylogenetic order)"
    else:
        row_positions = leaves_list(linkage(squareform(np.nan_to_num(squareform(
            pdist(clustered_matrix, "jaccard")))), "average"))
        ylabel = "433 human ORs (co-tuning order)"

    column_order = []
    bounds = []
    position = 0
    for c in display_order:
        members = np.where(labels == c)[0]
        if len(members) > 2:
            members = members[leaves_list(linkage(np.nan_to_num(
                pdist(clustered_matrix[:, members].T, "jaccard")), "average"))]
        column_order += members.tolist()
        position += len(members)
        bounds.append(position)
    column_order = np.array(column_order)
    shown = clustered_matrix[np.ix_(row_positions, column_order)]
    starts = [0] + bounds[:-1]

    fig = plt.figure(figsize=(12.4, 7.6))
    grid = fig.add_gridspec(2, 3, width_ratios=[0.11, 4, 0.11], height_ratios=[0.14, 4], hspace=0.028,
                            wspace=0.02, left=0.055, right=0.94, top=0.83, bottom=0.135)
    ax = fig.add_subplot(grid[1, 1])
    ax.imshow(shown, aspect="auto", cmap=ListedColormap(["#F4F1EA", "#2E4057"]), interpolation="nearest",
              rasterized=True)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_edgecolor("#bbb")
        spine.set_linewidth(0.6)
    for i in range(1, len(bounds)):
        ax.axvline(starts[i] - 0.5, color="white", lw=1.8)
    ax.set_xlabel(f"{len(clustered_ids)} odorants — co-tuning cluster (Jaccard, PAM, k={k})", fontsize=10.5,
                  labelpad=8, color="#8a8a86")
    ax.set_ylabel(ylabel, fontsize=10.5, color="#8a8a86")

    ax_top = fig.add_subplot(grid[0, 1], sharex=ax)
    ax_top.imshow(np.array([[matplotlib.colors.to_rgb(info[labels[j]]["color"]) for j in column_order]]),
                  aspect="auto", interpolation="nearest", extent=[0, len(column_order), 0, 1])
    ax_top.set_xlim(0, len(column_order))
    ax_top.set_ylim(0, 1)
    ax_top.set_xticks([])
    ax_top.set_yticks([])
    for spine in ax_top.spines.values():
        spine.set_visible(False)
    for i, c in enumerate(display_order):
        ax_top.text((starts[i] + bounds[i]) / 2, 1.55, SHORT_THEME.get(info[c]["name"], info[c]["name"][:10]),
                    ha="center", va="bottom", fontsize=9, color=info[c]["color"], fontweight="bold",
                    clip_on=False)

    ax_class = fig.add_subplot(grid[1, 0], sharey=ax)
    ax_class.imshow(np.array([[matplotlib.colors.to_rgb(
        CLASS_I_COLOUR if class_of[receptor_ids[j]] == "I" else CLASS_II_COLOUR)] for j in row_positions]),
        aspect="auto", interpolation="nearest")
    ax_class.set_xticks([])
    ax_class.set_yticks([])
    for spine in ax_class.spines.values():
        spine.set_visible(False)
    ax_class.set_xlabel("class", fontsize=7.5, color="#8a8a86", labelpad=4)

    ax_breadth = fig.add_subplot(grid[1, 2], sharey=ax)
    ax_breadth.imshow(clustered_matrix.sum(1)[row_positions].reshape(-1, 1), aspect="auto", cmap="Greys",
                      interpolation="nearest", rasterized=True)
    ax_breadth.set_xticks([])
    ax_breadth.set_yticks([])
    for spine in ax_breadth.spines.values():
        spine.set_visible(False)
    ax_breadth.set_xlabel("breadth", fontsize=7.5, color="#8a8a86", labelpad=4)

    ax.legend(handles=[Patch(fc=CLASS_I_COLOUR, label="Class I"), Patch(fc=CLASS_II_COLOUR, label="Class II"),
                       Patch(fc="#2E4057", label="binding"), Patch(fc="#F4F1EA", ec="#bbb", label="no binding")],
              loc="upper left", bbox_to_anchor=(0, -0.05), ncol=4, frameon=False, fontsize=9, handlelength=1.1,
              columnspacing=1.4, labelcolor="#8a8a86")
    fig.suptitle(title, fontsize=12.5, y=0.965, x=0.5, fontweight="bold", color=INK)
    fig.text(0.5, 0.905, subtitle, ha="center", fontsize=9.5, color="#a5a5a0")
    save(fig, path, show)


def pcoa_figure(clustered_matrix, labels, info, display_order, title, path, show):
    """The odorants on the first two principal coordinates of their Jaccard distances."""
    coordinates, eigenvalues = pcoa(jaccard_distance_between_columns(clustered_matrix))
    positive_total = eigenvalues[eigenvalues > 0].sum()
    first, second = 100 * eigenvalues[0] / positive_total, 100 * eigenvalues[1] / positive_total
    negative_share = 100 * (-eigenvalues[eigenvalues < 0].sum()) / (np.abs(eigenvalues).sum())

    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    for c in display_order:
        members = labels == c
        points = coordinates[members, :2]
        ax.scatter(points[:, 0], points[:, 1], s=15, color=info[c]["color"], alpha=0.75, edgecolor="none",
                   label=f'{info[c]["name"]} (n={info[c]["size"]})', zorder=3)
    ax.axhline(0, color="#e5e5e2", lw=0.6, zorder=0)
    ax.axvline(0, color="#e5e5e2", lw=0.6, zorder=0)
    ax.set_xlabel(f"PCo1 ({first:.1f}%)", fontsize=10.5, color=MUTED)
    ax.set_ylabel(f"PCo2 ({second:.1f}%)", fontsize=10.5, color=MUTED)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8.3, loc="best", labelcolor=INK)
    ax.set_title(title, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0.99, 0.01, f"non-Euclidean (neg. eigenvalue mass {negative_share:.0f}%)", transform=ax.transAxes,
            ha="right", va="bottom", fontsize=7, color="#b5b5b0")
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
    for c, tag, fold, q in rows:
        if previous is not None and c != previous:
            y -= 1
        ys.append(y)
        tag_labels.append(tag)
        colours.append(info[c]["color"])
        folds.append(fold)
        q_values.append(q)
        y -= 1
        previous = c
    ys = np.array(ys)
    folds = np.array(folds)

    fig, ax = plt.subplots(figsize=(8.6, 0.3 * (len(tag_labels) + len(display_order)) + 1))
    ax.barh(ys, folds, color=colours, alpha=0.92, height=0.74)
    ax.axvline(1, color=MUTED, ls="--", lw=1.0)
    x_max = folds.max()
    # Nudged down: asterisks sit high in the glyph box, so 'center' reads as floating.
    for y_value, fold, q in zip(ys, folds, q_values):
        ax.text(fold + x_max * 0.02, y_value - 0.12, significance_stars(q), va="center", ha="left", fontsize=10,
                color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels(tag_labels, fontsize=8.5)
    for tick_label, colour in zip(ax.get_yticklabels(), colours):
        tick_label.set_color(colour)
    ax.set_xlim(0, x_max * 1.14)
    ax.set_ylim(ys.min() - 1, ys.max() + 1)
    ax.set_xlabel("fold enrichment of odour descriptor  (observed / expected)\n"
                  "BH-corrected q:   * < 0.05      ** < 0.01      *** < 0.001", fontsize=10)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    save(fig, path, show)


# (key, header, x position in axes fraction, alignment)
_TABLE_COLUMNS = [("cluster", "cluster", 0.000, "left"), ("theme", "theme", 0.075, "left"),
                  ("tag", "odour tag", 0.290, "left"), ("obs", "in cluster", 0.470, "right"),
                  ("n", "cluster n", 0.570, "right"), ("tag_total", "tag total", 0.668, "right"),
                  ("exp", "expected", 0.762, "right"), ("fold", "fold", 0.838, "right"),
                  ("p", "p", 0.912, "right"), ("q", "q (BH)", 0.982, "right")]


def _format_p(x):
    return "%.1e" % x if x < 1e-3 else "%.3f" % x


def enrichment_table_rows(significant, info, display_order):
    """The significant enrichments as a table, clusters numbered C1-C6 in display order."""
    rows = []
    for i, c in enumerate(display_order):
        for row in significant[significant.cluster == c].sort_values("obs_exp", ascending=False).itertuples():
            rows.append(dict(cluster="C%d" % (i + 1), theme=info[c]["name"], tag=row.tag, obs=int(row.obs),
                             n=int(row.n), tag_total=int(row.tag_total), exp=float(row.exp),
                             fold=float(row.obs_exp), p=float(row.p), q=float(row.fdr),
                             stars=significance_stars(row.fdr), color=info[c]["color"]))
    return pd.DataFrame(rows)


def enrichment_table_figure(table, title, path, show):
    """The enrichment table drawn as a figure, one tinted band per cluster."""
    cell_text = {"cluster": lambda r: r.cluster, "theme": lambda r: SHORT_THEME.get(r.theme, r.theme),
                 "tag": lambda r: r.tag, "obs": lambda r: "%d" % r.obs, "n": lambda r: "%d" % r.n,
                 "tag_total": lambda r: "%d" % r.tag_total, "exp": lambda r: "%.1f" % r.exp,
                 "fold": lambda r: "%.1f" % r.fold, "p": lambda r: _format_p(r.p),
                 "q": lambda r: _format_p(r.q) + r.stars}
    row_height = 0.30
    fig, ax = plt.subplots(figsize=(11.4, row_height * (len(table) + 3.4)))
    ax.set_xlim(0, 1)
    ax.set_ylim(-len(table) - 0.5, 1.6)
    ax.axis("off")
    for key, header, x, align in _TABLE_COLUMNS:
        ax.text(x, 0.72, header, ha=align, va="center", fontsize=9, fontweight="bold", color=INK)
    ax.plot([0, 1], [0.2, 0.2], color=MUTED, lw=1.0)
    for i, row in enumerate(table.itertuples()):
        y = -i
        ax.add_patch(Rectangle((-0.008, y - 0.45), 1.016, 0.9, facecolor=row.color, alpha=0.13,
                               edgecolor="none", zorder=0))
        if i and row.cluster != table.cluster.iloc[i - 1]:
            ax.plot([0, 1], [y + 0.5, y + 0.5], color="#dcdcd6", lw=0.8, zorder=2)
        for key, header, x, align in _TABLE_COLUMNS:
            ax.text(x, y, cell_text[key](row), ha=align, va="center", fontsize=8.6, zorder=3,
                    color=row.color if key == "cluster" else TEXT,
                    fontweight="bold" if key in ("cluster", "fold") else "normal")
    ax.plot([0, 1], [-len(table) + 0.5 - 0.03, -len(table) + 0.5 - 0.03], color=MUTED, lw=1.0)
    ax.text(0, 1.45, title, ha="left", va="center", fontsize=12, fontweight="bold", color=INK)
    ax.text(1, -len(table) - 0.25, "hypergeometric, BH corrected over all cluster x tag tests   "
            "(*q<0.05  **q<0.01  ***q<0.001)", ha="right", va="center", fontsize=8, color=MUTED)
    save(fig, path, show)


def _cluster_tags(info, display_order):
    colours = [info[c]["color"] for c in display_order]
    tags = ["C%d" % (i + 1) for i in range(len(display_order))]
    names = [SHORT_THEME.get(info[c]["name"], info[c]["name"]) for c in display_order]
    return colours, tags, names


def cluster_descriptor_violins(descriptors, info, display_order, name, path, show):
    """One violin per cluster for each descriptor; Kruskal-Wallis across clusters, BH over the panel."""
    keys = [key for key, _ in CLUSTER_DESCRIPTORS]
    labels = [label for _, label in CLUSTER_DESCRIPTORS]
    colours, tags, names = _cluster_tags(info, display_order)
    data = {key: [descriptors.loc[descriptors.cluster == c, key].values for c in display_order] for key in keys}
    q_values = benjamini_hochberg(np.array([kruskal(*data[key])[1] for key in keys]))

    n_columns = 5
    n_rows = int(np.ceil(len(keys) / n_columns))
    fig, axes = plt.subplots(n_rows, n_columns, figsize=(2.9 * n_columns, 2.9 * n_rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, key, label, q in zip(axes, keys, labels, q_values):
        violins = ax.violinplot(data[key], positions=range(len(display_order)), widths=0.9, showmedians=False,
                                showextrema=False)
        for body, colour in zip(violins["bodies"], colours):
            body.set_facecolor(colour)
            body.set_alpha(0.85)
            body.set_edgecolor("none")
        for j, values in enumerate(data[key]):
            q1, median, q3 = np.percentile(values, [25, 50, 75])
            ax.vlines(j, q1, q3, color=TEXT, lw=1.5, zorder=4)
            ax.plot(j, median, "o", mfc="white", mec=TEXT, mew=1.0, ms=3.6, zorder=5)
        ax.set_title("%s   %s" % (label, significance_stars(q) or "ns"), fontsize=9.5, color=INK, pad=6)
        ax.set_xticks(range(len(display_order)))
        ax.set_xticklabels(tags, fontsize=8.5)
        for tick_label, colour in zip(ax.get_xticklabels(), colours):
            tick_label.set_color(colour)
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.tick_params(axis="y", labelsize=8)
        ax.tick_params(axis="x", length=0)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    for ax in axes[len(keys):]:
        ax.axis("off")
    fig.legend(handles=[Patch(facecolor=c, label="%s  %s" % (t, n)) for c, t, n in zip(colours, tags, names)],
               loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=len(display_order), frameon=False,
               fontsize=9, handlelength=1.1, handleheight=1.1, columnspacing=1.8)
    fig.suptitle("Chemical space of the activation clusters — %s" % name, fontsize=12, fontweight="bold",
                 color=INK, y=0.995)
    fig.text(0.5, 0.012, "violin = distribution, bar = IQR, dot = median   ·   "
             "Kruskal-Wallis across the %d clusters, BH:  * < 0.05   ** < 0.01   *** < 0.001" % len(display_order),
             ha="center", fontsize=8.5, color=MUTED)
    fig.tight_layout(rect=[0, 0.028, 1, 0.915])
    save(fig, path, show)


def cluster_functional_groups(descriptors, info, display_order, name, path, show):
    """Share of each cluster carrying each group; Fisher one cluster against the rest, BH over the grid."""
    groups = list(FUNCTIONAL_GROUPS)
    colours, tags, names = _cluster_tags(info, display_order)
    percent = np.zeros((len(groups), len(display_order)))
    p_values = np.ones_like(percent)
    for i, group in enumerate(groups):
        for j, c in enumerate(display_order):
            inside = descriptors.cluster == c
            carry_in = int(descriptors.loc[inside, group].sum())
            lack_in = int(inside.sum()) - carry_in
            carry_out = int(descriptors.loc[~inside, group].sum())
            lack_out = int((~inside).sum()) - carry_out
            percent[i, j] = 100 * carry_in / max(int(inside.sum()), 1)
            p_values[i, j] = fisher_exact([[carry_in, lack_in], [carry_out, lack_out]])[1]
    q_values = benjamini_hochberg(p_values.ravel()).reshape(p_values.shape)
    order = np.argsort(percent.mean(1))          # rarest group at the bottom
    groups = [groups[i] for i in order]
    percent = percent[order]
    q_values = q_values[order]

    ys = np.arange(len(groups))
    bar_height = 0.84 / len(display_order)
    x_max = percent.max()
    fig, ax = plt.subplots(figsize=(9.2, 0.92 * len(groups) + 1.7))
    for j, c in enumerate(display_order):
        offset = ((len(display_order) - 1) / 2 - j) * bar_height     # C1 at the top of each group
        ax.barh(ys + offset, percent[:, j], height=bar_height * 0.9, color=colours[j], alpha=0.92,
                label="%s  %s" % (tags[j], names[j]))
        for i in range(len(groups)):
            stars = significance_stars(q_values[i, j])
            if stars:
                ax.text(percent[i, j] + x_max * 0.012, ys[i] + offset - 0.06, stars, va="center", ha="left",
                        fontsize=7.5, color=MUTED)
    for i in ys[:-1]:
        ax.axhline(i + 0.5, color="#ecebe6", lw=0.8, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(groups, fontsize=9.5)
    ax.set_ylim(-0.5, len(groups) - 0.5)
    ax.set_xlim(0, x_max * 1.10)
    ax.set_xlabel("% of cluster carrying the group\n"
                  "Fisher one-vs-rest, BH:   * < 0.05      ** < 0.01      *** < 0.001", fontsize=10)
    ax.set_title("Functional groups by activation cluster — %s" % name, fontsize=12, fontweight="bold", color=INK)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", ncol=2, handlelength=1.2)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.tick_params(left=False)
    save(fig, path, show)


def cluster_effects(descriptors, info, display_order, name, path, show):
    """Cliff's delta of each cluster against all other odorants, per descriptor."""
    keys = [key for key, _ in CLUSTER_DESCRIPTORS]
    labels = [label for _, label in CLUSTER_DESCRIPTORS]
    colours, tags, names = _cluster_tags(info, display_order)
    delta = np.zeros((len(keys), len(display_order)))
    p_values = np.ones_like(delta)
    for i, key in enumerate(keys):
        for j, c in enumerate(display_order):
            inside = descriptors.loc[descriptors.cluster == c, key].values
            outside = descriptors.loc[descriptors.cluster != c, key].values
            delta[i, j] = cliffs_delta(inside, outside)
            try:
                p_values[i, j] = mannwhitneyu(inside, outside).pvalue
            except Exception:
                p_values[i, j] = np.nan
    q_values = benjamini_hochberg(np.nan_to_num(p_values.ravel(), nan=1.0)).reshape(p_values.shape)
    limit = np.abs(delta).max()

    fig, ax = plt.subplots(figsize=(1.15 * len(display_order) + 4.4, 0.55 * len(keys) + 2.0))
    image = ax.imshow(delta, cmap=DIVERGING, aspect="auto", vmin=-limit, vmax=limit)
    for i in range(len(keys)):
        for j in range(len(display_order)):
            ax.text(j, i, "%+.2f%s" % (delta[i, j], significance_stars(q_values[i, j])), ha="center",
                    va="center", fontsize=8, color="white" if abs(delta[i, j]) > 0.62 * limit else TEXT)
    ax.set_xticks(range(len(display_order)))
    ax.set_xticklabels(["%s\n%s" % (t, n) for t, n in zip(tags, names)], fontsize=8.5)
    for tick_label, colour in zip(ax.get_xticklabels(), colours):
        tick_label.set_color(colour)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xticks(np.arange(-.5, len(display_order), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(keys), 1), minor=True)
    ax.grid(which="minor", color="white", lw=1.6)
    ax.tick_params(which="minor", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    colourbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colourbar.set_label("Cliff's delta, cluster vs all other molecules", fontsize=9)
    colourbar.outline.set_visible(False)
    ax.set_title("What each activation cluster is made of — %s\n"
                 "Mann-Whitney one-vs-rest, BH:  * < 0.05   ** < 0.01   *** < 0.001" % name,
                 fontsize=11, fontweight="bold", color=INK)
    fig.tight_layout()
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


def _breadth_title(stats):
    shift, low, high = stats["shift_all"]
    return (f"Mann-Whitney $P$ = {stats['p_all']:.3g} {significance_stars(stats['p_all'])},   "
            f"Cliff's $\\delta$ = {stats['delta_all']:+.2f},   shift {shift:+.0f} ligands "
            f"[95% CI {low:+.0f}, {high:+.0f}]")


def breadth_histogram(stats, path, show=True):
    """Receptors per ligand-count bin, as % of each class. Bin width doubles; 0 has its own bar.

    Percent rather than counts, because on a shared count axis the 62 Class I receptors are
    a sliver next to 371 Class II. The counts are printed above the bars.
    """
    series = {"Class I": stats["class_I"], "Class II": stats["class_II"]}
    counts = {k: np.bincount(np.searchsorted(BREADTH_EDGES, v, side="right") - 1,
                             minlength=len(BREADTH_LABELS)) for k, v in series.items()}
    last = max(np.nonzero(c)[0].max() for c in counts.values()) + 1
    counts = {k: c[:last] for k, c in counts.items()}
    x = np.arange(last)
    percent = {k: 100 * c / len(series[k]) for k, c in counts.items()}
    legend_text = {k: f"{k}    n={len(v)}, median {np.median(v):g}, {(v == 0).sum()} orphans"
                   for k, v in series.items()}

    width = 0.40
    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    for offset, k in [(-width / 2, "Class I"), (width / 2, "Class II")]:
        ax.bar(x + offset, percent[k], width=width, color=CLASS_COLOURS[k], alpha=0.95, zorder=3,
               label=legend_text[k])
        for xi, (pct, n) in enumerate(zip(percent[k], counts[k])):
            if n:
                ax.text(xi + offset, pct + 0.4, str(n), ha="center", va="bottom", fontsize=7.5,
                        color=CLASS_COLOURS[k], zorder=4)
    ax.set_ylim(0, max(max(p) for p in percent.values()) * 1.24)
    ax.set_xticks(x)
    ax.set_xticklabels(BREADTH_LABELS[:last], fontsize=9)
    ax.set_xlim(-0.7, last - 0.3)
    ax.set_xlabel("ligands per receptor   (bin width doubles; 0 = orphan receptors)", fontsize=10)
    ax.set_ylabel("% of class", fontsize=10, color=MUTED)
    ax.tick_params(axis="y", labelsize=8.5)
    handles, labels = ax.get_legend_handles_labels()
    order = [labels.index(legend_text[k]) for k in ["Class I", "Class II"]]
    ax.legend([handles[i] for i in order], [labels[i] for i in order], frameon=False, fontsize=9,
              loc="upper right")
    ax.set_title(f"Class I receptors are more broadly tuned\n{_breadth_title(stats)}", fontsize=10, color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    ax.figure.text(0.5, -0.02, "bar height = % of that class;  number above each bar = receptors",
                   ha="center", fontsize=8.5, color=MUTED)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def breadth_strip(stats, path, show=True):
    """Every receptor as one point, with median and IQR, on a symlog axis so 0 has a position."""
    series = {"Class I": stats["class_I"], "Class II": stats["class_II"]}
    ticks = [0, 1, 3, 10, 30, 100, 300]
    fig, ax = plt.subplots(figsize=(5.4, 4.3))
    rng = np.random.default_rng(0)
    for i, (k, values) in enumerate(series.items()):
        jitter = i + rng.uniform(-0.17, 0.17, len(values))
        ax.scatter(jitter, values, s=14, color=CLASS_COLOURS[k], alpha=0.5, edgecolor="none", zorder=3)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        ax.plot([i - 0.33, i + 0.33], [median, median], color=INK, lw=2.6, zorder=5, solid_capstyle="round")
        ax.plot([i - 0.22, i + 0.22], [q1, q1], color=INK, lw=1.1, zorder=5)
        ax.plot([i - 0.22, i + 0.22], [q3, q3], color=INK, lw=1.1, zorder=5)
        ax.plot([i, i], [q1, q3], color=INK, lw=1.1, zorder=4)
        ax.annotate(f"median {median:g}", xy=(i + 0.33, median), xytext=(6, -3), textcoords="offset points",
                    fontsize=8.5, color=CLASS_COLOURS[k], fontweight="bold", va="center")
        ax.annotate(f"{(values == 0).sum()} orphan", xy=(i, 0), xytext=(0, -17), textcoords="offset points",
                    fontsize=7.5, color=MUTED, ha="center")
    ax.set_yscale("symlog", linthresh=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticks, fontsize=8.5)
    ax.set_ylim(-0.6, 900)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"{k}\nn={len(v)}" for k, v in series.items()], fontsize=9.5)
    for tick_label, k in zip(ax.get_xticklabels(), series):
        tick_label.set_color(CLASS_COLOURS[k])
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylabel("ligands per receptor", fontsize=10, color=MUTED)
    ax.set_title(f"Class I receptors are more broadly tuned\n"
                 f"Mann-Whitney $P$ = {stats['p_all']:.3g},  Cliff's $\\delta$ = {stats['delta_all']:+.2f}",
                 fontsize=9.5, color=INK)
    ax.tick_params(axis="x", length=0, pad=14)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


def _log_density(values, xs, bandwidth=0.42):
    """KDE of log10(ligands) over non-orphans, truncated at one ligand, scaled by the non-orphan share.

    Truncated rather than reflected: reflecting doubles the many receptors that bind exactly
    one ligand and plants a false peak at 1.
    """
    values = np.asarray(values, float)
    bound = values[values > 0]
    density = gaussian_kde(np.log10(bound), bw_method=bandwidth)(xs)
    return density / np.trapezoid(density, xs) * (len(bound) / len(values))


def breadth_density(stats, path, show=True):
    """Orphans as a bar at 0, the rest as a density on log10(ligands); bar plus curve is 1 per class."""
    from scipy.stats import mannwhitneyu as mwu

    class_I, class_II = stats["class_I"], stats["class_II"]
    u_statistic, _ = mwu(class_I, class_II)
    series = {"Class I": class_I, "Class II": class_II}
    bar_width = 0.15
    bar_x = -0.40
    xs = np.linspace(0.0, np.log10(max(class_I.max(), class_II.max())) * 1.06, 512)

    fig, ax = plt.subplots(figsize=(8.8, 5.0))
    y_max = 0
    for k, values in series.items():
        orphan_share = (values == 0).mean()
        ax.bar(bar_x, orphan_share / bar_width, width=bar_width, align="edge", color=CLASS_COLOURS[k], alpha=0.55,
               edgecolor=CLASS_COLOURS[k], lw=1.6, zorder=3)
        density = _log_density(values, xs)
        y_max = max(y_max, density.max(), orphan_share / bar_width)
        ax.fill_between(xs, density, color=CLASS_COLOURS[k], alpha=0.32, zorder=2, lw=0)
        ax.plot(xs, density, color=CLASS_COLOURS[k], lw=2.0, zorder=4,
                label=f"{k}    n={len(values)}, median {np.median(values):g}, {(values == 0).sum()} orphans")
    ax.set_ylim(0, y_max * 1.32)
    for k, values in series.items():
        median = float(np.median(values))
        ax.axvline(np.log10(median), color=CLASS_COLOURS[k], ls="--", lw=1.4, zorder=5)
        ax.annotate(f"median {median:g}", xy=(np.log10(median), ax.get_ylim()[1]), xytext=(4, -6),
                    textcoords="offset points", fontsize=8.5, color=CLASS_COLOURS[k], fontweight="bold",
                    va="top")
    ax.annotate("orphans", xy=(bar_x + bar_width / 2, y_max * 1.02), ha="center", va="bottom", fontsize=8.5,
                color=MUTED)
    ticks = [1, 2, 5, 10, 20, 50, 100, 200, 500]
    ax.set_xticks([bar_x + bar_width / 2] + list(np.log10(ticks)))
    ax.set_xticklabels([0] + ticks, fontsize=9)
    ax.set_xlim(bar_x - 0.12, xs[-1])
    ax.set_xlabel("ligands per receptor   (log scale; 0 is a separate category, off the scale)", fontsize=10)
    ax.set_ylabel("density", fontsize=10, color=MUTED)
    ax.tick_params(axis="y", labelsize=8.5)
    ax.grid(axis="y", color="#ecebe6", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=9, loc="upper right")
    shift, low, high = stats["shift_all"]
    ax.set_title(f"Class I receptors are more broadly tuned\n"
                 f"Mann-Whitney $U$ = {u_statistic:.0f},  $P$ = {stats['p_all']:.3g} "
                 f"{significance_stars(stats['p_all'])},   Cliff's $\\delta$ = {stats['delta_all']:+.2f},   "
                 f"shift {shift:+.0f} ligands [95% CI {low:+.0f}, {high:+.0f}]", fontsize=10, color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    fig.text(0.5, -0.02, "all 433 receptors: bar at 0 = that class's orphan fraction (area, not height), "
             "curve = the receptors with ≥1 ligand carrying the rest of the mass;  bar + curve = 1 for each class",
             ha="center", fontsize=8.5, color=MUTED)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------------------------
# What each class reads (Fig. 2c,d)
# ---------------------------------------------------------------------------------------------

def class_functional_groups(descriptors, name, path, show):
    """Share of each class set carrying each functional group.

    The stars compare Class I only against Class II only: Fisher per group, Benjamini-Hochberg
    over the groups shown. The shared set is drawn for context.
    """
    sets = {g: descriptors[descriptors.group == g] for g in CLASS_SETS}
    sizes = {g: len(sets[g]) for g in CLASS_SETS}
    groups = list(FUNCTIONAL_GROUPS)
    fraction = {g: [100 * sets[g][x].mean() if len(sets[g]) else np.nan for x in groups] for g in CLASS_SETS}
    p_values = []
    for x in groups:
        carry_I = int(sets["Class I only"][x].sum())
        carry_II = int(sets["Class II only"][x].sum())
        try:
            p_values.append(fisher_exact([[carry_I, sizes["Class I only"] - carry_I],
                                          [carry_II, sizes["Class II only"] - carry_II]])[1])
        except Exception:
            p_values.append(np.nan)
    q_values = benjamini_hochberg(np.nan_to_num(np.array(p_values), nan=1.0))
    order = np.argsort(fraction["Class I only"])
    groups = [groups[i] for i in order]
    q_values = [q_values[i] for i in order]
    fraction = {g: [fraction[g][i] for i in order] for g in CLASS_SETS}

    ys = np.arange(len(groups))
    bar_height = 0.26
    fig, ax = plt.subplots(figsize=(8.6, 0.72 * len(groups) + 1.2))
    for offset, g in zip([bar_height, 0, -bar_height], CLASS_SETS):
        ax.barh(ys + offset, fraction[g], height=bar_height, color=CLASS_SET_COLOURS[g], alpha=0.9, label=g)
    x_max = np.nanmax([np.nanmax(fraction[g]) for g in CLASS_SETS])
    for y, q in zip(ys, q_values):
        ax.text(x_max * 1.02, y, significance_stars(q) or "ns", va="center", fontsize=9, color=MUTED)
    ax.set_yticks(ys)
    ax.set_yticklabels(groups, fontsize=9)
    ax.set_xlabel("% of odorants containing group", fontsize=10)
    ax.set_xlim(0, x_max * 1.15)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")
    ax.set_title("Functional groups — %s   (significance: I-only vs II-only, BH)" % name, fontsize=12,
                 fontweight="bold", color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    save(fig, path, show)
    return pd.DataFrame({"group": groups, **{g: fraction[g] for g in CLASS_SETS}, "q": q_values})


CLASS_DESCRIPTORS = [("MolWt", "MW (Da)"), ("LogP", "cLogP"), ("TPSA", "TPSA (Å²)"), ("HBD", "H-bond donors"),
                     ("HBA", "H-bond acceptors"), ("AromaticRings", "aromatic rings"),
                     ("RotBonds", "rotatable bonds"), ("RingCount", "ring count"), ("FracCSP3", "fraction sp³"),
                     ("Heteroatoms", "heteroatoms")]


def class_descriptor_violins(descriptors, name, path, show):
    """Descriptor violins for the three class sets.

    The stars compare the two exclusive sets: Mann-Whitney, Benjamini-Hochberg over the ten descriptors.
    """
    sets = {g: descriptors[descriptors.group == g] for g in CLASS_SETS}
    sizes = {g: len(sets[g]) for g in CLASS_SETS}
    p_values = []
    for key, _ in CLASS_DESCRIPTORS:
        try:
            p_values.append(mannwhitneyu(sets["Class I only"][key].values, sets["Class II only"][key].values).pvalue)
        except Exception:
            p_values.append(np.nan)
    q_values = benjamini_hochberg(np.nan_to_num(np.array(p_values), nan=1.0))

    fig, axes = plt.subplots(2, 5, figsize=(16, 7))
    axes = axes.ravel()
    for ax, (key, label), q in zip(axes, CLASS_DESCRIPTORS, q_values):
        data = [sets[g][key].values for g in CLASS_SETS]
        parts = ax.violinplot(data, showmedians=True, widths=0.85)
        for body, g in zip(parts["bodies"], CLASS_SETS):
            body.set_facecolor(CLASS_SET_COLOURS[g])
            body.set_alpha(0.65)
            body.set_edgecolor("#333")
        for part in ["cmedians", "cmaxes", "cmins", "cbars"]:
            parts[part].set_color("#555")
            parts[part].set_linewidth(1.0)
        ax.set_title("%s\nI-only vs II-only  %s" % (label, significance_stars(q) or "ns"), fontsize=9.5, color=INK)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["I only", "shared", "II only"], fontsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)
    fig.suptitle("Chemical space by receptor-class bag — %s  ·  I only n=%d, shared n=%d, II only n=%d"
                 % (name, sizes["Class I only"], sizes["shared"], sizes["Class II only"]),
                 fontsize=13, fontweight="bold", y=1.0, color=INK)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save(fig, path, show)


CLIFF_DESCRIPTORS = [("MolWt", "MW (Da)"), ("LogP", "hydrophobicity (cLogP)"), ("TPSA", "polar surface (TPSA)"),
                     ("HBD", "H-bond donors"), ("HBA", "H-bond acceptors"), ("AromaticRings", "aromatic rings"),
                     ("RotBonds", "rotatable bonds"), ("RingCount", "ring count"),
                     ("FracCSP3", "fraction sp$^3$"), ("Heteroatoms", "heteroatoms")]


def cliffs_figure(descriptors, in_I, in_II, title, path, show):
    """Cliff's delta per descriptor between two disjoint sets, sorted by effect; BH over the panel.

    Only for disjoint sets: Mann-Whitney needs two independent samples.
    """
    set_I = descriptors[in_I]
    set_II = descriptors[in_II]
    keys = [key for key, _ in CLIFF_DESCRIPTORS]
    label_of = dict(CLIFF_DESCRIPTORS)
    effects, p_values = [], []
    for key in keys:
        effects.append(cliffs_delta(set_I[key].values, set_II[key].values))
        try:
            p_values.append(mannwhitneyu(set_I[key].values, set_II[key].values).pvalue)
        except Exception:
            p_values.append(np.nan)
    q_values = benjamini_hochberg(np.nan_to_num(np.array(p_values), nan=1.0))
    order = np.argsort(effects)
    effects = np.array(effects)[order]
    labels = [label_of[keys[i]] for i in order]
    q_values = q_values[order]

    fig, ax = plt.subplots(figsize=(8, 0.5 * len(labels) + 1.2))
    ax.barh(range(len(labels)), effects, color=[CLASS_I_COLOUR if e > 0 else CLASS_II_COLOUR for e in effects],
            alpha=0.9)
    ax.axvline(0, color="#c9c9c4", lw=0.8)
    x_range = max(abs(effects.min()), abs(effects.max()), 0.05)
    for i, (effect, q) in enumerate(zip(effects, q_values)):
        ax.text(effect + 0.03 * x_range * (1 if effect >= 0 else -1), i, significance_stars(q) or "ns",
                va="center", ha="left" if effect >= 0 else "right", fontsize=8.5, color=MUTED)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(-x_range * 1.3, x_range * 1.3)
    ax.set_xlabel("effect size (Cliff delta)    $\\leftarrow$ Class II only higher   |   "
                  "Class I only higher $\\rightarrow$", fontsize=9.5)
    ax.set_title(title, fontsize=12, fontweight="bold", color=INK)
    for side in ["top", "right"]:
        ax.spines[side].set_visible(False)
    save(fig, path, show)
    return labels, effects, q_values


# Rows of the fold-change panel: (key, label, how the fold change is taken, print format).
#   mean     ratio of means       counts, fractions, prevalences (the median of most is 0)
#   median   ratio of medians     strictly positive continuous quantities
#   antilog  10 ** (median diff)  quantities already on a log scale (cLogP, log vapour pressure)
FOLD_BLOCKS = [
    ("water affinity", [
        ("WaterAff", "water affinity  1/$P$ = 10$^{-\\mathrm{cLogP}}$", "antilog", "%.1e"),
        ("TPSA", "polar surface (TPSA, Å$^2$)", "median", "%.1f"),
        ("HBD", "H-bond donors (-OH, -NH)", "mean", "%.2f"),
        ("HBA", "H-bond acceptors", "mean", "%.2f"),
        ("OxygenAtoms", "oxygen atoms", "mean", "%.2f")]),
    ("size and volatility", [
        ("MolWt", "molecular weight (Da)", "median", "%.0f"),
        ("Tb", "boiling point (Joback, K)", "median", "%.0f"),
        ("VP", "vapour pressure at 25 °C (atm)", "antilog", "%.1e"),
        ("RotBonds", "flexibility (rotatable bonds)", "mean", "%.2f")]),
    ("shape", [
        ("AromaticRings", "aromatic rings per molecule", "mean", "%.2f"),
        ("RingCount", "rings", "mean", "%.2f"),
        ("FracCSP3", "saturation (fraction sp$^3$)", "mean", "%.2f")]),
    ("functional groups — % of the set carrying it", [
        (g, g, "mean", "%.0f%%") for g in ["carboxylic acid", "alcohol", "ester", "aldehyde", "ketone", "ether",
                                            "aromatic ring", "amine", "sulfur"]]),
]


def fold_change_figure(table, in_I, in_II, title, subtitle, name_I, name_II, figure_path, table_path,
                       show, blocks=None, n_permutations=5000, n_bootstrap=4000):
    """Fold change per property, Class I set over Class II set, on a log2 axis.

    Works for overlapping sets (the two full class bags) and disjoint ones. When the sets
    overlap, the shared odorants enter both sides whatever the labels are, so the null
    reassigns only the exclusive labels and the bootstrap draws the shared block once for
    both sides. A group present in one set and absent from the other is an infinite fold
    change, drawn as a bar running off the axis.

    Returns
    -------
    DataFrame
        Fold change, 95% bootstrap CI, the two group values, permutation p and q, Cliff's delta.
    """
    blocks = FOLD_BLOCKS if blocks is None else blocks
    keys = [key for _, items in blocks for key, *_ in items]
    modes = [mode for _, items in blocks for _, _, mode, _ in items]
    labels = [label for _, items in blocks for _, label, _, _ in items]
    formats = [fmt for _, items in blocks for _, _, _, fmt in items]
    percent_keys = {key for _, items in blocks for key, _, _, fmt in items if fmt.endswith("%%")}
    use_mean = np.array([mode == "mean" for mode in modes])
    use_antilog = np.array([mode == "antilog" for mode in modes])
    values = table[keys].values.astype(float)
    shared = in_I & in_II
    only_I = in_I & ~in_II
    only_II = in_II & ~in_I

    def group_value(rows):
        with np.errstate(all="ignore"):
            return np.where(use_mean, np.nanmean(rows, 0), np.nanmedian(rows, 0))

    def fold(a, b):
        with np.errstate(all="ignore"):
            return np.where(use_antilog, 10.0 ** (a - b), a / b)

    value_I, value_II = group_value(values[in_I]), group_value(values[in_II])
    with np.errstate(all="ignore"):
        fold_change = fold(value_I, value_II)
        log_fold = np.log2(fold_change)

    rng = np.random.default_rng(0)
    exclusive = np.where(only_I | only_II)[0]
    n_only_I = int(only_I.sum())
    hits = np.zeros(len(keys))
    for _ in range(n_permutations):
        permuted = rng.permutation(exclusive)
        side_a = np.zeros(len(table), bool)
        side_b = np.zeros(len(table), bool)
        side_a[permuted[:n_only_I]] = True
        side_b[permuted[n_only_I:]] = True
        with np.errstate(all="ignore"):
            permuted_fold = np.abs(np.log2(fold(group_value(values[shared | side_a]),
                                                group_value(values[shared | side_b]))))
        hits += np.nan_to_num(permuted_fold, nan=-1.0) >= np.abs(log_fold) - 1e-12
    p_values = (hits + 1) / (n_permutations + 1)
    q_values = benjamini_hochberg(p_values)

    index_I, index_II, index_shared = np.where(only_I)[0], np.where(only_II)[0], np.where(shared)[0]
    bootstrap = np.empty((n_bootstrap, len(keys)))
    for i in range(n_bootstrap):
        shared_draw = values[rng.choice(index_shared, len(index_shared))] if len(index_shared) else values[:0]
        with np.errstate(all="ignore"):
            bootstrap[i] = np.log2(fold(
                group_value(np.vstack([values[rng.choice(index_I, len(index_I))], shared_draw])),
                group_value(np.vstack([values[rng.choice(index_II, len(index_II))], shared_draw]))))
    ci_low, ci_high = np.nanpercentile(np.where(np.isfinite(bootstrap), bootstrap, np.nan), [2.5, 97.5], axis=0)

    # A property with no value in either set (0/0) is dropped rather than drawn empty.
    defined = ~np.isnan(fold_change)
    if not defined.all():
        print("  %s: dropped %s — no value in either set"
              % (figure_path.stem, ", ".join(np.array(keys)[~defined])))
    ys, rows, block_positions = [], [], []
    y = 0.0
    position = 0
    for block_name, items in blocks:
        indices = [position + t for t in range(len(items)) if defined[position + t]]
        position += len(items)
        if not indices:
            continue
        block_positions.append((y + 0.78, block_name))
        for t in indices:
            rows.append(t)
            ys.append(y)
            y -= 1
        y -= 1.5
    ys = np.array(ys)
    rows = np.array(rows)

    # Margins in inches, so a six-row panel keeps the same title block and row pitch as a
    # twenty-one-row one.
    top_inches, bottom_inches = 1.44, 1.56
    height = max(5.0, top_inches + bottom_inches + 0.267 * (len(rows) + 1.5 * len(block_positions)))

    def inches(v):
        return v / height

    fig = plt.figure(figsize=(12.8, height))
    grid = fig.add_gridspec(1, 2, width_ratios=[3.05, 0.95], wspace=0.02, left=0.30, right=0.965,
                            top=1 - inches(top_inches), bottom=inches(bottom_inches))
    ax = fig.add_subplot(grid[0, 0])
    ax_values = fig.add_subplot(grid[0, 1], sharey=ax)
    blend = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    span = np.concatenate([log_fold[rows], ci_low[rows], ci_high[rows]])
    x_limit = float(np.nanmax(np.abs(span[np.isfinite(span)]))) + 0.95
    ax.axvline(0, color="#9a9a96", lw=1.0, zorder=1)
    for i, y_value in zip(rows, ys):
        stars = significance_stars(q_values[i]) or "ns"
        colour = CLASS_I_COLOUR if log_fold[i] >= 0 else CLASS_II_COLOUR
        if np.isfinite(log_fold[i]):
            ax.barh(y_value, log_fold[i], height=0.62, color=colour, alpha=0.92, zorder=3)
            if np.isfinite(ci_low[i]) and np.isfinite(ci_high[i]):
                ax.plot([ci_low[i], ci_high[i]], [y_value, y_value], color=TEXT, lw=1.0, zorder=4,
                        solid_capstyle="butt")
                for end in (ci_low[i], ci_high[i]):
                    ax.plot([end, end], [y_value - 0.15, y_value + 0.15], color=TEXT, lw=1.0, zorder=4)
            side = 1 if log_fold[i] >= 0 else -1
            x_text = max(ci_high[i], log_fold[i]) if side > 0 else min(ci_low[i], log_fold[i])
            if not np.isfinite(x_text):
                x_text = log_fold[i]
            text = ("%.2f×  %s" if fold_change[i] < 10 else "%.1f×  %s") % (fold_change[i], stars)
        else:
            side = 1 if log_fold[i] > 0 else -1
            x_text = side * (x_limit - 0.60)
            ax.barh(y_value, x_text, height=0.62, color=colour, alpha=0.92, zorder=3)
            ax.plot([x_text], [y_value], marker=">" if side > 0 else "<", ms=7, color=colour, zorder=4)
            text = "$\\infty$  %s" % stars
        ax.text(x_text + 0.11 * side, y_value, text, va="center", ha="left" if side > 0 else "right",
                fontsize=8.6, zorder=5, color=TEXT if stars != "ns" else "#a5a5a0")
    # Ticks mirrored about 1x, powers of two only, so "twice" and "half" sit equally far out.
    ticks = sorted({t for b in [1, 2, 4, 8, 16, 32, 64] for t in (b, 1.0 / b) if abs(np.log2(t)) <= x_limit})
    ax.set_xticks([np.log2(t) for t in ticks])
    ax.set_xticklabels(["%.3g×" % t for t in ticks], fontsize=9)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(ys.min() - 1.35, ys.max() + 1.45)
    ax.set_yticks(ys)
    ax.set_yticklabels([labels[i] for i in rows], fontsize=9.6)
    ax.tick_params(axis="y", length=0)
    for side_name in ["top", "right", "left"]:
        ax.spines[side_name].set_visible(False)
    ax.spines["bottom"].set_color("#c9c9c4")
    for y_value, block_name in block_positions:
        ax.text(-0.285, y_value, block_name.upper(), transform=blend, ha="left", va="center", fontsize=8.6,
                color=MUTED, fontweight="bold", clip_on=False)
        ax.plot([-0.285, 1.0], [y_value + 0.42] * 2, transform=blend, color="#e3e1db", lw=0.9, zorder=1,
                clip_on=False)
    ax.set_xlabel("fold change   (%s ÷ %s, log$_2$ axis)" % (name_I, name_II), fontsize=10.5, labelpad=7)
    ax.text(-x_limit * 0.52, ys.min() - 0.80, "$\\leftarrow$  more in %s" % name_II, ha="center", va="center",
            fontsize=9.6, color=CLASS_II_COLOUR)
    ax.text(x_limit * 0.52, ys.min() - 0.80, "more in %s  $\\rightarrow$" % name_I, ha="center", va="center",
            fontsize=9.6, color=CLASS_I_COLOUR)
    ax_values.set_xlim(0, 1)
    ax_values.axis("off")
    ax_values.text(0.56, ys.max() + 1.35, "group value", ha="center", fontsize=9, color=MUTED)
    ax_values.text(0.32, ys.max() + 0.75, "Class I", ha="center", fontsize=9.4, color=CLASS_I_COLOUR,
                   fontweight="bold")
    ax_values.text(0.80, ys.max() + 0.75, "Class II", ha="center", fontsize=9.4, color=CLASS_II_COLOUR,
                   fontweight="bold")
    for i, y_value in zip(rows, ys):
        a, b = value_I[i], value_II[i]
        if modes[i] == "antilog":
            a, b = 10.0 ** a, 10.0 ** b
        if keys[i] in percent_keys:
            a, b = 100 * a, 100 * b
        ax_values.text(0.32, y_value, formats[i] % a, ha="center", va="center", fontsize=8.8, color=TEXT)
        ax_values.text(0.80, y_value, formats[i] % b, ha="center", va="center", fontsize=8.8, color=TEXT)
    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, y=1 - inches(0.187))
    fig.text(0.5, 1 - inches(0.728), subtitle, ha="center", fontsize=9.6, color=MUTED, linespacing=1.5)
    fig.text(0.5, inches(0.312), "bars = ratio of group medians (means for counts, fractions and "
             "group prevalences%s)   ·   whiskers = 95%% bootstrap CI\n"
             "permutation of the set labels, %d draws, BH:  * < 0.05   ** < 0.01   *** < 0.001"
             % ("; $10^{\\Delta}$ for the rows that are already logarithms" if use_antilog.any() else "",
                n_permutations), ha="center", fontsize=8.6, color=MUTED)
    save(fig, figure_path, show)

    display_I = [10.0 ** v if mode == "antilog" else v for v, mode in zip(value_I, modes)]
    display_II = [10.0 ** v if mode == "antilog" else v for v, mode in zip(value_II, modes)]
    effects = pd.DataFrame(dict(
        property=keys, fold_change_I_over_II=fold_change, ci_lo=2 ** ci_low, ci_hi=2 ** ci_high,
        class_I=display_I, class_II=display_II, summary=modes, perm_p=p_values, perm_q=q_values,
        cliffs_delta=[cliffs_delta(values[in_I, i][np.isfinite(values[in_I, i])],
                                   values[in_II, i][np.isfinite(values[in_II, i])]) for i in range(len(keys))]))
    effects.to_csv(table_path, index=False)
    print("  %-42s %s" % (figure_path.stem, "  ".join("%s %.2fx%s" % (k, f, significance_stars(q))
                                                     for k, f, q in zip(keys, fold_change, q_values) if q < 0.05)))
    return effects


def tag_percent_figure(shown, n_tagged_I, n_tagged_II, title, subtitle, name_I, name_II, path, show,
                       min_pct=8.0, test=False):
    """Grouped bars: share of each set's tagged odorants carrying each odour descriptor."""
    ys = np.arange(len(shown))[::-1]
    bar_height = 0.38
    height = max(4.6, 3.1 + 0.40 * len(shown))
    fig, ax = plt.subplots(figsize=(9.6, height))
    ax.barh(ys + bar_height / 2, shown.pct_I, height=bar_height, color=CLASS_I_COLOUR, alpha=0.92,
            label="%s  (n=%d tagged)" % (name_I, n_tagged_I))
    ax.barh(ys - bar_height / 2, shown.pct_II, height=bar_height, color=CLASS_II_COLOUR, alpha=0.92,
            label="%s  (n=%d tagged)" % (name_II, n_tagged_II))
    x_max = max(shown.pct_I.max(), shown.pct_II.max())
    for y, row in zip(ys, shown.itertuples()):
        ax.text(row.pct_I + x_max * 0.012, y + bar_height / 2, "%.0f%%" % row.pct_I, va="center", fontsize=8,
                color=TEXT)
        ax.text(row.pct_II + x_max * 0.012, y - bar_height / 2, "%.0f%%" % row.pct_II, va="center", fontsize=8,
                color=TEXT)
    for y in ys[:-1]:
        ax.axhline(y - 0.5, color="#f0eee9", lw=0.8, zorder=0)
    # Stars in their own column: the test is on the descriptor, not on either bar.
    if test:
        for y, stars in zip(ys, shown.stars):
            ax.text(x_max * 1.13, y, stars, ha="center", va="center", fontsize=8.6,
                    color=TEXT if stars != "ns" else "#b9b9b4")
    ax.set_yticks(ys)
    ax.set_yticklabels(shown.tag, fontsize=9.5)
    ax.set_ylim(ys.min() - 0.7, ys.max() + 0.7)
    ax.set_xlim(0, x_max * (1.20 if test else 1.10))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("% of the set's tagged molecules carrying the tag", fontsize=10)
    for side in ["top", "right", "left"]:
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c9c9c4")
    ax.legend(frameon=False, fontsize=9, handlelength=1.2, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.004))
    fig.suptitle(title, fontsize=12.5, fontweight="bold", color=INK, y=1 - 0.18 / height)
    fig.text(0.5, 1 - 0.60 / height, subtitle, ha="center", fontsize=9.4, color=MUTED)
    note = ("Fisher one tag at a time, BH over the %d tags drawn:  * < 0.05   ** < 0.01   *** < 0.001"
            % len(shown)) if test else "percentages only — the two sets overlap, so a per-tag test would not be valid"
    fig.text(0.5, 0.18 / height, "sorted by the gap  ·  a tag is drawn if it reaches %g%% of one set\n%s"
             % (min_pct, note), ha="center", fontsize=8.4, color=MUTED, linespacing=1.5)
    fig.subplots_adjust(top=1 - 1.20 / height, bottom=1.15 / height, left=0.17, right=0.97)
    save(fig, path, show)
