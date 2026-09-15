"""
Main figure, ancestral analysis.

Panel a is left blank for the tree + activation-pattern heatmap, which are placed
by hand in Inkscape.  Panels b-d are rebuilt from Tables/ so the whole figure comes
out as one vector PDF with live text.

All three data panels use the no-overlay density fill (the 06_* tables), so the
figure is internally consistent: ancestral cut 0.8875, extant cut 0.8947.
"""
from pathlib import Path

import matplotlib as mpl
mpl.use("pdf")
import matplotlib.pyplot as plt
import pandas as pd

HERE   = Path(__file__).resolve().parent
TABLES = HERE / "Tables"
FIGS   = HERE / "Figures"

# palette, taken from the notebooks
CLASS1  = "#7f1734"
CLASS2  = "#557c99"
COMB    = "#b08a3e"
SINGLE  = "#c9c9c4"
INK     = "#3a3a38"
MUTED   = "#8a8a86"
FAINT   = "#d6d6d1"

mpl.rcParams.update({
    "pdf.fonttype":       42,     # TrueType, so Inkscape keeps the text editable
    "svg.fonttype":       "none",
    "font.family":        "sans-serif",
    "font.sans-serif":    ["DejaVu Sans", "Helvetica", "Arial"],
    "font.size":          7,
    "axes.labelsize":     7,
    "axes.titlesize":     7.5,
    "xtick.labelsize":    6.5,
    "ytick.labelsize":    6.5,
    "legend.fontsize":    6.5,
    "axes.edgecolor":     MUTED,
    "axes.labelcolor":    INK,
    "text.color":         INK,
    "xtick.color":        MUTED,
    "ytick.color":        MUTED,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "xtick.major.size":   2.5,
    "ytick.major.size":   2.5,
    "lines.linewidth":    1.1,
})

MM = 1 / 25.4
FIG_W, FIG_H = 180 * MM, 200 * MM
fig = plt.figure(figsize=(FIG_W, FIG_H))


def panel_label(x, y, letter):
    fig.text(x, y, letter, fontsize=9, fontweight="bold", color=INK,
             ha="left", va="top")


def despine(ax, keep=("left", "bottom")):
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(side in keep)


# ----------------------------------------------------------------------------
# a  -- reserved for the tree and the activation-pattern heatmap
# ----------------------------------------------------------------------------
A_BOTTOM, A_HEIGHT = 0.455, 0.520
ax_a = fig.add_axes([0.055, A_BOTTOM, 0.925, A_HEIGHT])
ax_a.set_xticks([]); ax_a.set_yticks([])
for side in ax_a.spines.values():
    side.set_visible(True)
    side.set_color(FAINT)
    side.set_linestyle((0, (4, 4)))
    side.set_linewidth(0.6)
ax_a.text(0.5, 0.5,
          "panel a — place tree + activation-pattern heatmap here",
          ha="center", va="center", fontsize=7, color=FAINT, style="italic")
panel_label(0.02, 0.985, "a")


# ----------------------------------------------------------------------------
# b  -- the first three stages, calibrated cut
# ----------------------------------------------------------------------------
ROW_BOTTOM, ROW_HEIGHT = 0.105, 0.275
ax_b = fig.add_axes([0.070, ROW_BOTTOM, 0.220, ROW_HEIGHT])

stages = pd.read_csv(TABLES / "07_stages.csv")
cal = stages[stages["panel"] == "calibrated"].reset_index(drop=True)

x = range(len(cal))
ax_b.bar(x, cal["labelled_line"], color=SINGLE,
         label="labelled line (1 receptor)", width=0.62, linewidth=0)
ax_b.bar(x, cal["combinatorial"], bottom=cal["labelled_line"], color=COMB,
         label="combinatorial ($\\geq$2 receptors)", width=0.62, linewidth=0)

for i, row in cal.iterrows():
    ax_b.text(i, row["encoded"] + 1.6, f"{row['pct_combinatorial']:.0f}%",
              ha="center", va="bottom", fontsize=6.5, fontweight="bold", color=INK)
    ax_b.text(i, row["labelled_line"] / 2, f"{int(row['labelled_line'])}",
              ha="center", va="center", fontsize=6, color="#6e6e6a")
    if row["combinatorial"] > 0:
        ax_b.text(i, row["labelled_line"] + row["combinatorial"] / 2,
                  f"{int(row['combinatorial'])}", ha="center", va="center",
                  fontsize=6, color="white")

ax_b.set_xticks(list(x))
ax_b.set_xticklabels(["1\nnode 0", "2\nnodes 1,2", "3\nnodes 3,4,5"], fontsize=6)
ax_b.set_xlabel("receptors in the repertoire", labelpad=2)
ax_b.set_ylabel("odorants encoded", labelpad=2)
ax_b.set_ylim(0, 80)
ax_b.set_title("a combinatorial code appears\nat the first duplication",
               loc="left", pad=5, color=INK)
ax_b.legend(frameon=False, loc="upper left", handlelength=1.0,
            handleheight=0.9, borderpad=0, labelspacing=0.3,
            bbox_to_anchor=(-0.02, 1.005))
despine(ax_b)
panel_label(0.020, ROW_BOTTOM + ROW_HEIGHT + 0.070, "b")


# ----------------------------------------------------------------------------
# c  -- receptors per class, both time axes
# ----------------------------------------------------------------------------
ax_c = fig.add_axes([0.410, ROW_BOTTOM, 0.220, ROW_HEIGHT])

ultra = pd.read_csv(TABLES / "06_trajectory.csv")
raw   = pd.read_csv(TABLES / "06_trajectory_raw_axis.csv")

ax_c.plot(raw["t"],   raw["n_class_II"], color=CLASS2, ls=":", lw=1.0)
ax_c.plot(raw["t"],   raw["n_class_I"],  color=CLASS1, ls=":", lw=1.0)
ax_c.plot(ultra["t"], ultra["n_class_II"], color=CLASS2, lw=1.2, label="Class II")
ax_c.plot(ultra["t"], ultra["n_class_I"],  color=CLASS1, lw=1.2, label="Class I")

ax_c.text(1.015, 371, "371", color=CLASS2, fontsize=6.5, va="center", fontweight="bold")
ax_c.text(1.015, 62,  "62",  color=CLASS1, fontsize=6.5, va="center", fontweight="bold")

ax_c.set_xlim(0, 1.0)
ax_c.set_ylim(-8, 400)
ax_c.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax_c.set_xticklabels(["node 0\n(origin)", "0.25", "0.5", "0.75", "human\n(today)"])
ax_c.set_xlabel("relative time, root to tip", labelpad=2)
ax_c.set_ylabel("receptor lineages", labelpad=2)
ax_c.set_title("Class I saturates,\nClass II keeps expanding",
               loc="left", pad=5, color=INK)

leg_c = ax_c.legend(frameon=False, loc="upper left", handlelength=1.3,
                    borderpad=0, labelspacing=0.3, bbox_to_anchor=(-0.02, 1.02))
ax_c.add_artist(leg_c)
ax_c.text(0.03, 0.60, "solid   ultrametric axis\ndotted  raw substitutions",
          transform=ax_c.transAxes, fontsize=5.8, color=MUTED, va="top",
          linespacing=1.4)
despine(ax_c)
panel_label(0.360, ROW_BOTTOM + ROW_HEIGHT + 0.070, "c")


# ----------------------------------------------------------------------------
# d  -- combinatoriality against the density-forced baseline
# ----------------------------------------------------------------------------
ax_d = fig.add_axes([0.755, ROW_BOTTOM, 0.220, ROW_HEIGHT])

null = pd.read_csv(TABLES / "06_independence_null.csv").drop_duplicates(subset="n", keep="last")

ax_d.plot(null["n"], null["forced"] * 100, color=SINGLE, ls="--", lw=1.1,
          label="forced by density alone")
ax_d.plot(null["n"], null["observed"] * 100, color=INK, lw=1.2, label="observed")

ax_d.set_xscale("log")
ax_d.set_xlim(2, 500)
ax_d.set_ylim(-4, 108)
ax_d.set_xlabel("receptor lineages (log)", labelpad=2)
ax_d.set_ylabel("% of detected odorants\nwith $\\geq$2 detectors", labelpad=2)
ax_d.set_title("the rise is arithmetic —\nthe code stays $\\it{below}$ it",
               loc="left", pad=5, color=INK)
ax_d.legend(frameon=False, loc="lower right", handlelength=1.6,
            borderpad=0, labelspacing=0.3, bbox_to_anchor=(1.0, 0.01))
despine(ax_d)
panel_label(0.705, ROW_BOTTOM + ROW_HEIGHT + 0.070, "d")


# ----------------------------------------------------------------------------
fig.text(0.070, 0.030,
         "All panels use the density fill with no experimental overlay "
         "(ancestral cut 0.8875, extant cut 0.8947).",
         fontsize=5.5, color=MUTED, va="bottom")
fig.text(0.070, 0.014,
         "Node numbers follow panel a:  node 0 = node_1;  nodes 1,2 = node_2, node_3;  "
         "nodes 3,4,5 = node_4, node_5, node_6.",
         fontsize=5.5, color=MUTED, va="bottom")

out_pdf = FIGS / "MainFigure_ancestral.pdf"
out_svg = FIGS / "MainFigure_ancestral.svg"
fig.savefig(out_pdf)
fig.savefig(out_svg)
print("wrote", out_pdf)
print("wrote", out_svg)
