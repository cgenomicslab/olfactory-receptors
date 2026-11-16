"""Drawing functions for the ancestral panels of Fig. 3 and Extended Data Figs. 5-6.

Every function takes tables written by `Downstream_Analysis/ancestral_receptors/a01`-`a07`
and an output path, draws one printed panel and saves it as SVG. Nothing here
computes a result.

Node numbers on every panel are the paper's: tree `node_k` is printed as k - 1.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

from plotting_functions import FIGSIZE

CLASS1_COL, CLASS2_COL, COMMON_COL = "#7f1734", "#557c99", "#5F6A72"
GREY = "#8a8a86"        # numbers and axis text on the main-figure panels
MID = "#9a9a96"
YLAB = "#000000"        # y-axis titles on the main-figure panels
BOTH_COL, GOLD = "#c9c4bb", "#ae7121"


def save(fig, path, show):
    """Write the figure as an SVG at its figure size, then show it inline or close it."""
    fig.savefig(path)
    if show:
        plt.show()
    else:
        plt.close(fig)
MM = 1 / 25.4
FS = 8.0
PATTERN_ORDER = ["100", "111", "011", "010", "110", "001", "101"]


def _main_figure_style(draw):
    """Draw a main-figure panel with 8 pt text throughout, ticks included."""
    import functools

    @functools.wraps(draw)
    def wrapped(*args, **kwargs):
        with plt.rc_context({"font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS,
                             "ytick.labelsize": FS, "text.color": GREY}):
            return draw(*args, **kwargs)
    return wrapped


def figure_node(name):
    """Paper number of a tree node: "node_7" or 7 -> 6."""
    if isinstance(name, str):
        name = int(name.removeprefix("node_"))
    return int(name) - 1


def _frame(ax):
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GREY)
    ax.tick_params(length=2, color=GREY, labelcolor=GREY, pad=1.5)


def _time_axis(ax):
    ax.set_xlim(0, 1)
    ax.set_xticks([0, .25, .5, .75, 1])
    ax.set_xticklabels(["node 0\n(origin)", "0.25", "0.5", "0.75", "human\n(today)"],
                       fontsize=FS, color=GREY, linespacing=1.05)
    ax.set_xlabel("relative time, root to tip (ultrametric)", fontsize=FS, color=GREY, labelpad=1)


# ------------------------------------------------------------------------- Figure 3

@_main_figure_style
def node_repertoires(patterns, path, show=True):
    """Fig. 3b: every ligand of nodes 0-2 as a column, one row per node.

    `patterns` is ligand_patterns.csv; the pattern string reads [class I, common,
    class II]. Columns are grouped by pattern; a filled cell means the node binds it.
    """
    patterns = patterns.assign(pattern=patterns.pattern.astype(str).str.zfill(3))   # read back as integers
    order = sorted(patterns.itertuples(), key=lambda r: (PATTERN_ORDER.index(r.pattern), r.SMILES))
    common = tuple((np.array(to_rgb(CLASS1_COL)) + np.array(to_rgb(CLASS2_COL))) / 2)
    rows = [("1", 0, CLASS1_COL), ("0", 1, common), ("2", 2, CLASS2_COL)]

    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for y, (label, bit, colour) in enumerate(rows):
        for x, r in enumerate(order):
            filled = r.pattern[bit] == "1"
            ax.add_patch(plt.Rectangle((x, y + 0.12), 1, 0.76, facecolor=colour if filled else "#f2f2f2",
                                       edgecolor="#dddddd", linewidth=0.4))
    ax.set_xlim(0, len(order)); ax.set_ylim(len(rows), 0)
    ax.set_yticks([0.5, 1.5, 2.5]); ax.set_yticklabels([r[0] for r in rows], fontsize=FS, color=GREY)
    ax.set_xticks([])
    ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, path, show)


@_main_figure_style
def first_split(group_sizes, path, show=True):
    """Fig. 3c: how the odorants of nodes 0-2 split between the class ancestors."""
    gs = group_sizes.set_index("group").n_ligands
    seg = [("Class I only", int(gs.class1_side), CLASS1_COL), ("both", int(gs.shared), "#9a8f80"),
           ("Class II only", int(gs.class2_side), CLASS2_COL),
           ("dropped by both", int(gs.common_only), "#d9d6cf")]
    tot = sum(v for _, v, _ in seg)
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    left = 0
    for i, (lab, v, c) in enumerate(seg):
        ax.barh(0, v, left=left, height=.46, color=c, alpha=.92, zorder=3)
        ax.text(left + v / 2, .34, "%d" % v, ha="center", va="bottom", fontsize=FS, color=GREY)
        yl = -.34 if i % 2 == 0 else -.92
        if i % 2:
            ax.plot([left + v / 2] * 2, [-.26, -.78], color=GREY, lw=.7, zorder=2)
        ax.text(left + v / 2, yl, lab, ha="center", va="top", fontsize=FS, color=GREY)
        left += v
    ax.set_xlim(-.4, tot + .4); ax.set_ylim(-2.05, 1.25); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("the %d odorants read by the first two receptors" % tot, fontsize=FS, color=GREY, pad=3)
    save(fig, path, show)


@_main_figure_style
def gain_loss_ladder(topology, path, show=True):
    """Fig. 3d: ligands gained and lost on the five branches below node 0."""
    br = topology[topology.node.isin([2, 3, 4, 5, 6])].copy()
    br["lab"] = ["%d → %d" % (figure_node(p), figure_node(n)) for p, n in zip(br.parent, br.node)]
    br["col"] = [CLASS1_COL if c == "Class_I" else CLASS2_COL for c in br.OR_class]
    br = br.iloc[::-1]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    y = np.arange(len(br))
    ax.barh(y, br.n_gained, height=.58, color=br.col, alpha=.92, zorder=3)
    ax.barh(y, -br.n_lost, height=.58, color=br.col, alpha=.40, zorder=3)
    ax.axvline(0, color=MID, lw=.8, zorder=4)
    xm = max(br.n_gained.max(), br.n_lost.max()) * 1.72
    for yi, r in zip(y, br.itertuples()):
        ax.text(r.n_gained + xm * .03, yi, "+%d" % r.n_gained, va="center", ha="left", fontsize=FS, color=r.col)
        ax.text(-r.n_lost - xm * .03, yi, "−%d" % r.n_lost, va="center", ha="right", fontsize=FS, color=GREY)
        ax.text(xm * .99, yi, "%d→%d" % (r.n_ligands + r.n_lost - r.n_gained, r.n_ligands),
                ha="right", va="center", fontsize=FS, color=GREY)
    ax.set_yticks(y); ax.set_yticklabels(br.lab, fontsize=FS, color=GREY)
    ax.set_xlim(-xm, xm); ax.set_ylim(-.95, len(br) - .30); ax.set_xticks([])
    ax.tick_params(axis="y", length=0, pad=1.5)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.text(-xm * .45, -.80, "← ligands lost", ha="center", fontsize=FS, color=GREY)
    ax.text(xm * .45, -.80, "ligands gained →", ha="center", fontsize=FS, color=GREY)
    save(fig, path, show)


@_main_figure_style
def acids_down_lineage(functional_groups, path, show=True):
    """Fig. 3e: % of each node's repertoire with a carboxylic acid (solid) or aromatic ring (open)."""
    fg = functional_groups.set_index("functional_group")
    order = [1, 2, 4, 5, 3, 6]
    cls = {1: "m", 2: "I", 4: "I", 5: "I", 3: "II", 6: "II"}
    acid = [fg.loc["carboxylic_acid", str(n)] for n in order]
    arom = [fg.loc["aromatic_ring", str(n)] for n in order]
    col = [MID if cls[n] == "m" else (CLASS1_COL if cls[n] == "I" else CLASS2_COL) for n in order]
    x = np.arange(len(order)); w = 0.34
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.bar(x - w / 2, acid, width=w, color=col, alpha=.92, zorder=3)
    ax.bar(x + w / 2, arom, width=w, facecolor="none", edgecolor=col, lw=.9, zorder=3)
    for xi, v in zip(x - w / 2, acid):
        ax.text(xi, v + 3, "%.0f" % v, ha="center", va="bottom", fontsize=FS, color=GREY)
    for xi, v in zip(x + w / 2, arom):
        ax.text(xi, v + 3, "%.0f" % v, ha="center", va="bottom", fontsize=FS, color=GREY)
    ax.set_ylim(0, 132); ax.set_yticks([0, 50, 100])
    ax.set_yticklabels(["0", "50", "100"], fontsize=FS, color=GREY)
    ax.set_ylabel("% of ligands", fontsize=FS, color=YLAB, labelpad=2)
    ax.set_xticks(x); ax.set_xticklabels([str(figure_node(n)) for n in order], fontsize=FS, color=GREY)
    ax.set_xlabel("node        Class I: 1, 3, 4        Class II: 2, 5", fontsize=FS, color=GREY, labelpad=1)
    ax.tick_params(axis="y", length=2, color=GREY); ax.tick_params(axis="x", length=0, pad=1.5)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    for sp in ["left", "bottom"]:
        ax.spines[sp].set_color(GREY)
    key = [plt.Rectangle((0, 0), 1, 1, facecolor=GREY, edgecolor="none"),
           plt.Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=GREY, lw=.9)]
    lg = ax.legend(key, ["carboxylic acid", "aromatic ring"], frameon=False, fontsize=FS, loc="upper center",
                   ncol=2, handlelength=.9, handleheight=.8, borderaxespad=.05, columnspacing=1.2,
                   handletextpad=.4)
    for t in lg.get_texts():
        t.set_color(GREY)
    save(fig, path, show)


def _edge_class(edges):
    """Class of each edge's child, found by walking up the edge table to node_2 or node_3."""
    parent_of = dict(zip(edges.child, edges.parent))

    def walk(n):
        seen = set()
        while True:
            if n == "node_2":
                return "Class_I"
            if n == "node_3":
                return "Class_II"
            if n not in parent_of or n in seen:
                return None
            seen.add(n)
            n = parent_of[n]

    return [walk(c) for c in edges.child]


@_main_figure_style
def half_life(edges, fit, path, show=True):
    """Fig. 3f: parent-child Jaccard against branch length, with the exponential fit of 04."""
    e = edges.copy()
    e["OR_class"] = _edge_class(e)
    f = fit.iloc[0]
    a, b, c, half = f.a, f.b, f.c, f.half_life
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    n_i, n_ii = int((e.OR_class == "Class_I").sum()), int((e.OR_class == "Class_II").sum())
    for k, col in [("Class_II", CLASS2_COL), ("Class_I", CLASS1_COL)]:
        g = e[e.OR_class == k]
        ax.scatter(g.branch_len, g.similarity, s=2.2, color=col, alpha=.40, lw=0, zorder=2)
    xs = np.linspace(0, e.branch_len.max(), 300)
    ax.plot(xs, a * np.exp(-b * xs) + c, color="#4a4a46", lw=1.4, zorder=5)
    ax.axhline(c, color=GREY, lw=.8, ls=(0, (2, 2)), zorder=3)
    ax.axvline(half, color=GREY, lw=.8, ls=(0, (2, 2)), zorder=3)
    ax.text(half + .03, 1.10, "half-life %.3f" % half, fontsize=FS, color=GREY, va="top")
    ax.text(e.branch_len.max() * .99, c + .035, "asymptote %.2f" % c, fontsize=FS, color=GREY, ha="right")
    ax.set_xlabel("branch length (substitutions/site)", fontsize=FS, color=GREY, labelpad=1)
    ax.set_ylabel("parent–child Jaccard similarity", fontsize=FS, color=YLAB, labelpad=2)
    ax.set_ylim(-.02, 1.22); ax.set_xlim(-.02, e.branch_len.max() * 1.02); _frame(ax)
    key = [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS1_COL, markersize=3.4, lw=0),
           plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=CLASS2_COL, markersize=3.4, lw=0)]
    lg = ax.legend(key, ["Class I  (n=%d)" % n_i, "Class II  (n=%d)" % n_ii], frameon=False, fontsize=FS,
                   loc="upper right", handlelength=.7, handletextpad=.4, borderaxespad=.15, labelspacing=.25)
    for t in lg.get_texts():
        t.set_color(GREY)
    ax.set_title("y = %.2f·exp(−%.1f d) + %.2f    R² = %.2f" % (a, b, c, f.r2),
                 fontsize=FS, color=GREY, pad=3)
    save(fig, path, show)


@_main_figure_style
def detecting_class(trajectory, path, show=True):
    """Fig. 3g: odorants detected by class I only, both classes, class II only."""
    tr = trajectory
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.stackplot(tr.t, tr.only_I, tr.both, tr.only_II, colors=[CLASS1_COL, BOTH_COL, CLASS2_COL], lw=0, zorder=2)
    ax.plot(tr.t, tr.cum_detected, color=GREY, lw=.9, ls=(0, (3, 2)), zorder=4)
    last = tr.iloc[-1]
    ax.text(1.01, last.only_I / 2, "%d" % last.only_I, color=CLASS1_COL, fontsize=FS, va="center")
    ax.text(1.01, last.only_I + last.both / 2, "%d" % last.both, color=GREY, fontsize=FS, va="center")
    ax.text(1.01, last.only_I + last.both + last.only_II / 2, "%d" % last.only_II, color=CLASS2_COL,
            fontsize=FS, va="center")
    ax.text(1.01, last.cum_detected + 18, "%d" % last.n_detected, color=GREY, fontsize=FS, va="center")
    h = [plt.Rectangle((0, 0), 1, 1, color=k) for k in [CLASS1_COL, BOTH_COL, CLASS2_COL]]
    h.append(plt.Line2D([0], [0], color=GREY, lw=.9, ls=(0, (3, 2))))
    ax.legend(h, ["Class I only", "both classes", "Class II only", "all encoded"], frameon=False, fontsize=FS,
              loc="upper left", handlelength=.9, handleheight=.8, borderaxespad=.15, labelspacing=.25)
    ax.set_ylabel("odorants detected", fontsize=FS, color=YLAB, labelpad=2)
    ax.set_ylim(0, 780); _time_axis(ax); _frame(ax)
    ax.set_title("odorants, by detecting class", fontsize=FS, color=GREY, pad=3)
    save(fig, path, show)


@_main_figure_style
def single_vs_combinatorial(trajectory, path, show=True):
    """Fig. 3h: odorants read by one receptor against odorants read by two or more."""
    tr = trajectory
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.fill_between(tr.t, 0, tr.n_combinatorial, color=GOLD, alpha=.22, lw=0, zorder=2)
    ax.plot(tr.t, tr.n_combinatorial, color=GOLD, lw=1.2, zorder=4)
    ax.plot(tr.t, tr.n_single, color="#4a4a46", lw=1.2, zorder=4)
    ax.text(1.01, tr.n_combinatorial.iloc[-1], "%d" % tr.n_combinatorial.iloc[-1], color=GOLD, fontsize=FS,
            va="center")
    ax.text(1.01, tr.n_single.iloc[-1], "%d" % tr.n_single.iloc[-1], color=GREY, fontsize=FS, va="center")
    ax.legend([plt.Line2D([0], [0], color=GOLD, lw=1.2), plt.Line2D([0], [0], color="#4a4a46", lw=1.2)],
              ["combinatorial (≥2 receptors)", "single detector"], frameon=False, fontsize=FS,
              loc="upper left", handlelength=1.1, borderaxespad=.15, labelspacing=.25)
    ax.set_ylabel("odorants detected", fontsize=FS, color=YLAB, labelpad=2)
    ax.set_ylim(0, 780); _time_axis(ax); _frame(ax)
    ax.set_title("single detector vs combinatorial", fontsize=FS, color=GREY, pad=3)
    save(fig, path, show)


@_main_figure_style
def receptors_per_class(trajectory, trajectory_raw, path, show=True):
    """Fig. 3i: receptor lineages per class, ultrametric axis (solid) and raw axis (dotted)."""
    tr, raw = trajectory, trajectory_raw
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.plot(tr.t, tr.n_class_II, color=CLASS2_COL, lw=1.2, zorder=4)
    ax.plot(tr.t, tr.n_class_I, color=CLASS1_COL, lw=1.2, zorder=4)
    ax.plot(raw.t, raw.n_class_II, color=CLASS2_COL, lw=.8, ls=(0, (1.5, 1.5)), zorder=3)
    ax.plot(raw.t, raw.n_class_I, color=CLASS1_COL, lw=.8, ls=(0, (1.5, 1.5)), zorder=3)
    ax.text(1.01, tr.n_class_II.iloc[-1], "%d" % tr.n_class_II.iloc[-1], color=CLASS2_COL, fontsize=FS, va="center")
    ax.text(1.01, tr.n_class_I.iloc[-1], "%d" % tr.n_class_I.iloc[-1], color=CLASS1_COL, fontsize=FS, va="center")
    ax.legend([plt.Line2D([0], [0], color=CLASS2_COL, lw=1.2), plt.Line2D([0], [0], color=CLASS1_COL, lw=1.2),
               plt.Line2D([0], [0], color=GREY, lw=1.2), plt.Line2D([0], [0], color=GREY, lw=.8, ls=(0, (1.5, 1.5)))],
              ["Class II", "Class I", "solid: ultrametric", "dotted: raw substitutions"],
              frameon=False, fontsize=FS, loc="upper left", handlelength=1.1, borderaxespad=.15, labelspacing=.25)
    ax.set_ylabel("receptors in the repertoire", fontsize=FS, color=YLAB, labelpad=2)
    ax.set_ylim(0, 430); _time_axis(ax); _frame(ax)
    ax.set_title("receptors per class", fontsize=FS, color=GREY, pad=3)
    save(fig, path, show)


# ------------------------------------------------------------- Extended Data Fig. 5

ANC_COL, EXT_COL = "#3a3a38", "#a8a8a3"   # ancestor vs extant: greys, not the class colours

# Text sizes of the Extended Data panels, matched to Fig. 3: titles 8.5 pt, axis labels 8 pt,
# tick labels 7 pt, legends 7 pt. Numbers and notes written on a panel are 6.5-7 pt.
ED_TEXT = {"axes.titlesize": 8, "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
           "legend.fontsize": 7}


def _extended_data_style(draw):
    """Draw an Extended Data panel with the text sizes in ED_TEXT."""
    import functools

    @functools.wraps(draw)
    def wrapped(*args, **kwargs):
        with plt.rc_context(ED_TEXT):
            return draw(*args, **kwargs)
    return wrapped


OUTLINE = "#4d4d4d"      # violin outline
INNER = "#333333"        # quartile bar and whiskers inside a violin


def _points(ax, x, values, colour, width):
    """The values of one group as jittered points around x."""
    jitter = np.random.default_rng(0).uniform(-width, width, len(values))
    ax.scatter(x + jitter, values, s=7, color=colour, alpha=0.55, linewidths=0, zorder=2)


def box_with_points(ax, data, colours):
    """One open box per group in the group's colour, over the jittered values.

    Box: quartiles and median. Whiskers: the furthest value within 1.5 interquartile ranges.
    Groups sit at x = 1, 2, ... as in `Axes.boxplot`.
    """
    for x, (values, colour) in enumerate(zip(data, colours), start=1):
        _points(ax, x, values, colour, width=0.18)
        ax.boxplot(values, positions=[x], widths=0.5, showfliers=False, showcaps=False,
                   boxprops=dict(color=colour, linewidth=1.0), whiskerprops=dict(color=colour, linewidth=1.0),
                   medianprops=dict(color=colour, linewidth=1.8), zorder=3)
    ax.grid(axis="y", color="#e5e5e2", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.4, len(data) + 0.6)


def violin(ax, data, colours):
    """One filled violin per group, with the quartiles as a thick bar, the whiskers as a thin line and the median as a white dot.

    The density is a Gaussian kernel estimate drawn two bandwidths past the smallest and largest
    value, so the tips taper. A group with fewer than three values, or with one repeated value,
    gets the bar and dot only. Groups sit at x = 1, 2, ...
    """
    from scipy.stats import gaussian_kde

    for x, (values, colour) in enumerate(zip(data, colours), start=1):
        values = np.asarray(values, dtype=float)
        if len(values) >= 3 and values.std() > 0:
            kernel = gaussian_kde(values)
            bandwidth = kernel.factor * values.std(ddof=1)
            grid = np.linspace(values.min() - 2 * bandwidth, values.max() + 2 * bandwidth, 200)
            density = kernel(grid)
            half_width = 0.40 * density / density.max()
            ax.fill_betweenx(grid, x - half_width, x + half_width, facecolor=colour, edgecolor=OUTLINE,
                             linewidth=0.9, zorder=3)
        q1, median, q3 = np.percentile(values, [25, 50, 75])
        low = values[values >= q1 - 1.5 * (q3 - q1)].min()
        high = values[values <= q3 + 1.5 * (q3 - q1)].max()
        ax.plot([x, x], [low, high], color=INNER, linewidth=0.9, solid_capstyle="butt", zorder=4)
        ax.plot([x, x], [q1, q3], color=INNER, linewidth=3.0, solid_capstyle="butt", zorder=4)
        ax.plot([x], [median], "o", color="white", markersize=2.2, zorder=5)
    ax.grid(axis="y", color="#d9d9d5", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.set_xlim(0.4, len(data) + 0.6)


@_extended_data_style
def ancestor_vs_extant(decay, stats, path, show=True):
    """ED5a: ligands per node, ancestors against extant receptors."""
    s = stats.set_index("statistic").value
    data = [decay.loc[decay.kind == k, "n_ligands"].values for k in ("ancestor", "extant")]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    violin(ax, data, [ANC_COL, EXT_COL])
    ax.set_xticks([1, 2]); ax.set_xticklabels(["ancestors", "extant"])
    ax.set_ylabel("ligands bound")
    ax.set_title(f"median {s.median_ancestor:.0f} vs {s.median_extant:.0f}   MWU p = {s.mannwhitney_p:.2g}")
    save(fig, path, show)


@_extended_data_style
def repertoire_turnover(turnover, path, show=True):
    """ED5b: ligands retained, lost and gained on the two edges out of node 0."""
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    x = np.arange(len(turnover)); w = 0.26
    for i, (col, colr) in enumerate([("retained", "#5F6A72"), ("lost", "#8A2F1D"), ("gained", "#355E3B")]):
        ax.bar(x + (i - 1) * w, turnover[col], w, label=col, color=colr, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(["node " + e.replace("->", "→") for e in turnover.edge_fig])
    ax.set_ylabel("ligands"); ax.legend(frameon=False)
    ax.set_title("Repertoire turnover from the common ancestor")
    save(fig, path, show)


@_extended_data_style
def ligands_vs_root_distance(decay, stats, path, show=True):
    """ED5c: ligands bound against patristic distance from the root, every node."""
    s = stats.set_index("statistic").value
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for kind, colr, marker in [("extant", EXT_COL, "o"), ("ancestor", ANC_COL, "^")]:
        sub = decay[decay.kind == kind]
        ax.scatter(sub.distance, sub.n_ligands, s=14, alpha=0.6, color=colr,
                   marker=marker, edgecolor="none", label=f"{kind} (n={len(sub)})")
    xs = np.linspace(decay.distance.min(), decay.distance.max(), 100)
    ax.plot(xs, s.slope * xs + s.intercept, color="#8a8a86", lw=1.2, ls="--")
    ax.set_xlabel("patristic distance from root")
    ax.set_ylabel("ligands bound (density fill)")
    ax.set_title(f"Pearson r = {s.pearson_r:+.3f}, p = {s.pearson_p:.2g};  "
                 f"Spearman ρ = {s.spearman_r:+.3f}, p = {s.spearman_p:.2g}\n"
                 f"slope = {s.slope:+.1f} ligands per unit distance", fontsize=8)
    ax.legend(frameon=False)
    save(fig, path, show)


@_extended_data_style
def similarity_by_child_type(edges, fit, path, show=True):
    """ED5d: parent-child Jaccard against branch length, extant and ancestral children."""
    f = fit.iloc[0]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for kind, colr, mk in [("extant", EXT_COL, "o"), ("ancestor", ANC_COL, "^")]:
        sub = edges[edges.kind == kind]
        ax.scatter(sub.branch_len, sub.similarity, s=12, alpha=0.5, color=colr,
                   marker=mk, edgecolor="none", label=f"{kind} (n={len(sub)})")
    ax.set_xlabel("branch length (substitutions/site)")
    ax.set_ylabel("parent-child Jaccard similarity")
    ax.set_title(f"Spearman r = {f.spearman_r:+.3f}, p = {f.spearman_p:.1e}")
    ax.legend(frameon=False)
    save(fig, path, show)


@_extended_data_style
def similarity_by_length_tertile(edges, path, show=True):
    """ED5d: parent-child Jaccard in the short, mid and long thirds of branch length."""
    tertile = pd.qcut(edges.branch_len, 3, labels=["short", "mid", "long"])
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    data = [edges.loc[tertile == t, "similarity"].values for t in ["short", "mid", "long"]]
    violin(ax, data, ["#5F6A72"] * 3)
    ax.set_xticks([1, 2, 3]); ax.set_xticklabels(["short", "mid", "long"])
    ax.set_ylabel("Jaccard similarity"); ax.set_xlabel("branch-length tertile")
    ax.set_title("  ".join(f"{t} {np.median(d):.2f}" for t, d in zip(["short", "mid", "long"], data)))
    save(fig, path, show)


# ------------------------------------------------------------- Extended Data Fig. 6

@_extended_data_style
def gain_loss(topology, path, show=True):
    """ED6a: ligands gained and lost against the immediate parent, ten nodes."""
    gl = topology.dropna(subset=["n_gained"])
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    x = np.arange(len(gl))
    ax.bar(x - 0.2, gl.n_gained, 0.38, color="#355E3B", label="gained", alpha=0.9)
    ax.bar(x + 0.2, -gl.n_lost, 0.38, color="#8A2F1D", label="lost", alpha=0.9)
    ax.axhline(0, color="#c9c9c4", lw=0.9)
    ax.set_xticks(x)
    # ten branch names do not fit side by side; slant them, anchored under their bar
    ax.set_xticklabels([f"{figure_node(r.parent)} → {figure_node(r.node)}" for r in gl.itertuples()],
                       rotation=45, ha="right", rotation_mode="anchor")
    ax.set_ylabel("ligands"); ax.legend(frameon=False)
    ax.set_title("Turnover along each edge")
    save(fig, path, show)


@_extended_data_style
def functional_group_heatmap(functional_groups, path, show=True):
    """ED6b: % of each node's repertoire carrying each functional group."""
    prev = functional_groups.set_index("functional_group")
    labels = [str(figure_node(int(c))) for c in prev.columns]
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    im = ax.imshow(prev.values, cmap="Purples", aspect="auto", vmin=0)
    ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels)
    ax.set_xlabel("node")
    ax.set_yticks(range(len(prev))); ax.set_yticklabels(prev.index)
    cut = prev.values.max() * 0.6
    for i in range(prev.shape[0]):
        for j in range(prev.shape[1]):
            v = prev.values[i, j]
            if v >= 1:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v > cut else "#333")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label="% of repertoire")
    ax.set_title("Functional-group prevalence per node")
    save(fig, path, show)


@_extended_data_style
def shift_vectors(plane, orthogonality, path, show=True):
    """ED6c: the two duplication shift vectors in the plane they span."""
    shifts = plane[plane.kind == "shift"].set_index("item")
    loads = plane[plane.kind == "descriptor"]
    a1 = shifts.loc["dup1", ["x", "y"]].values.astype(float)
    a2 = shifts.loc["dup2", ["x", "y"]].values.astype(float)
    obs = float(orthogonality.observed_angle.iloc[0])
    lim = max(np.linalg.norm(a1), np.linalg.norm(a2)) * 1.35
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    th1, th2 = np.arctan2(a1[1], a1[0]), np.arctan2(a2[1], a2[0])
    sweep = (th2 - th1 + np.pi) % (2 * np.pi) - np.pi
    arc = th1 + np.linspace(0, sweep, 100)
    r = lim * 0.28
    ax.plot(r * np.cos(arc), r * np.sin(arc), color="#999", lw=0.9)
    mid = th1 + sweep / 2
    ax.text(r * 1.22 * np.cos(mid), r * 1.22 * np.sin(mid), f"{obs:.0f}°",
            ha="center", va="center", fontsize=7, fontweight="bold", color="#3a3a38")
    for a, col, lab in [(a1, CLASS1_COL, "dup 1  → acids"), (a2, CLASS2_COL, "dup 2  → aromatics")]:
        ax.annotate("", xy=a, xytext=(0, 0), zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.4, shrinkA=0, shrinkB=0))
        u = a / np.linalg.norm(a)
        ax.text(a[0] + u[0] * lim * 0.06, a[1] + u[1] * lim * 0.06, lab, color=col,
                fontsize=7, fontweight="bold", va="center",
                ha="right" if u[0] < -0.3 else "left" if u[0] > 0.3 else "center")
    for row in loads.itertuples():
        lx, ly = row.x * lim * 0.62, row.y * lim * 0.62
        ax.plot([0, lx], [0, ly], color="#c8c8c3", lw=0.7, zorder=1)
    # Each label sits just past the tip of its ray, on the side the ray points to. Rays that
    # point the same way end close together, so on each side the labels are taken from the
    # bottom up and each is lifted, if needed, to one line above the label below it.
    line = lim * 0.075
    for side in [-1, 1]:
        tips = sorted([(row.y * lim * 0.62, row.x * lim * 0.62, row.item) for row in loads.itertuples()
                       if np.sign(row.x) == side])
        previous = -np.inf
        for ly, lx, item in tips:
            ty = max(ly * 1.1, previous + line)
            ax.text(lx * 1.1 + side * lim * 0.05, ty, item, fontsize=7, color="#999",
                    ha="right" if side < 0 else "left", va="center", zorder=1)
            previous = ty
    ax.scatter([0], [0], s=26, facecolor="white", edgecolor="#555", linewidth=1.0, zorder=5)
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim * 0.45, lim); ax.set_aspect("equal")
    ax.axhline(0, color="#e5e5e0", lw=0.6, zorder=0); ax.axvline(0, color="#e5e5e0", lw=0.6, zorder=0)
    ax.set_title("shift in 13-descriptor space, from each parent to what it gained", fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, path, show)


@_extended_data_style
def orthogonality_null(null_draws, orthogonality, path, show=True):
    """ED6d: the angle between random gain sets of the same sizes, against the observed angle."""
    null = null_draws.angle.values
    obs = float(orthogonality.observed_angle.iloc[0])
    n_close = int((np.abs(null - 90) <= abs(obs - 90)).sum())
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.hist(null, bins=40, color="#d8d8d3", linewidth=0)
    ax.axvline(90, color="#999", lw=0.8, ls=":")
    ax.axvline(obs, color="#3a3a38", lw=1.8)
    ax.text(obs - 2, ax.get_ylim()[1] * 0.94, f"observed\n{obs:.0f}°",
            ha="right", va="top", fontsize=7, color="#3a3a38", fontweight="bold")
    ax.text(float(np.median(null)), ax.get_ylim()[1] * 0.94, f"null median {np.median(null):.0f}°",
            ha="center", va="top", fontsize=7, color="#666")
    ax.set_xlabel("angle between the two duplication shifts (deg)", fontsize=8)
    ax.set_ylabel("random gain-sets", fontsize=8)
    ax.set_title(f"{n_close} of {len(null)} random draws come as close to 90°", fontsize=8)
    save(fig, path, show)


@_extended_data_style
def pca_chemspace(coordinates, variance, orthogonality, crosschecks, path, show=True):
    """ED6e: PCA of the odorant panel with the two gain sets and their parent centroids."""
    pc = coordinates
    evr = variance.set_index("component").explained
    obs = float(orthogonality.observed_angle.iloc[0])
    null_median = float(orthogonality.null_median.iloc[0])
    permanova_p = float(crosschecks.loc[crosschecks.check.str.startswith("PERMANOVA"), "P"].iloc[0])
    fig, ax = plt.subplots(figsize=FIGSIZE, layout="constrained")
    ax.scatter(pc.PC1, pc.PC2, s=5, color="#e0e0db", linewidth=0, zorder=1, label=f"odorant panel (n={len(pc)})")
    for key, col, name in [("dup1", CLASS1_COL, "branch 0 → 1  gains (acids)"),
                           ("dup2", CLASS2_COL, "branch 2 → 5  gains (aromatics)")]:
        gained = pc[pc[f"{key}_gained"]]
        ax.scatter(gained.PC1, gained.PC2, s=20, color=col, linewidth=0, zorder=3, label=name)
        par = pc.loc[pc[f"{key}_parent"], ["PC1", "PC2"]].mean().values
        ax.annotate("", xy=gained[["PC1", "PC2"]].mean().values, xytext=par, zorder=4,
                    arrowprops=dict(arrowstyle="-|>", color=col, lw=2.0, shrinkA=0, shrinkB=0))
        ax.scatter(*par, s=40, facecolor="white", edgecolor=col, linewidth=1.4, zorder=5)
    # a handful of very large molecules stretch PC1; clip the view, not the data
    xlo, xhi = np.percentile(pc.PC1, [0.5, 99.0])
    ylo, yhi = np.percentile(pc.PC2, [0.5, 99.5])
    pad_x, pad_y = 0.10 * (xhi - xlo), 0.10 * (yhi - ylo)
    ax.set_xlim(xlo - pad_x, xhi + pad_x); ax.set_ylim(ylo - pad_y, yhi + pad_y)
    n_out = int(((pc.PC1 > xhi + pad_x) | (pc.PC1 < xlo - pad_x) |
                 (pc.PC2 > yhi + pad_y) | (pc.PC2 < ylo - pad_y)).sum())
    ax.set_xlabel(f"PC1  ({100 * evr.PC1:.0f}%)  — size and complexity", fontsize=8)
    ax.set_ylabel(f"PC2  ({100 * evr.PC2:.0f}%)  — lipophilicity", fontsize=8)
    ax.set_title("the duplications moved into different regions of chemical space", fontsize=8)
    # the open circle and the arrow are keyed in the legend, so the statistics block below stays
    # short enough to clear the points in the lower right
    handles, labels = ax.get_legend_handles_labels()
    handles += [plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
                           markeredgecolor="#6b6b66", markeredgewidth=1.2, markersize=4.5),
                plt.Line2D([0], [0], color="#6b6b66", lw=1.6)]
    labels += ["parent repertoire centroid", "shift to the gained molecules"]
    ax.legend(handles, labels, frameon=False, fontsize=7, loc="upper left", handletextpad=0.5,
              borderpad=0.2, labelspacing=0.3)
    ax.text(0.99, 0.02,
            f"13-D angle between shifts {obs:.0f}° (null median {null_median:.0f}°)\n"
            f"PERMANOVA P = {permanova_p:.3f}\n"
            f"{n_out} panel molecules fall outside the view",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=7, color="#6b6b66", linespacing=1.3)
    save(fig, path, show)
