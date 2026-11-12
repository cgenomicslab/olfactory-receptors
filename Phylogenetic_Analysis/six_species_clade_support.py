"""Support of the main clades of the six-species tree across alignments (Figure 1d).

Reads data/ReferenceTree/clade_support.csv: one row per clade and tree, with the SH-aLRT on the
maximum-likelihood tree and the percentage of the 1,000 ultrafast bootstrap trees that contain
the clade after amphioxus is removed. The table is computed from the trees of all six
alignments, which are in the six-species archive on Zenodo.

The figure has the size of the other plots of the repository, 4 x 3 in. The panel is drawn in
millimetres at 1:1 (88 x 42 mm) and centred in it.

Run from Phylogenetic_Analysis/:  python six_species_clade_support.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Circle, FancyBboxPatch, Polygon

matplotlib.rcParams["svg.fonttype"] = "none"
matplotlib.rcParams["font.family"] = "sans-serif"
matplotlib.rcParams["font.sans-serif"] = ["Helvetica", "Arial", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"]

COLOUR = {
    "human class I":  "#7f1734",
    "human class II": "#557c99",
    "zebrafish":      "#2A9D8F",
    "hagfish":        "#ab7383",
    "catshark":       "#C56E4F",
    "amphioxus":      "#C4A01D",
}
INK, MUTED, BRANCH = "#222222", "#6b6b6b", "#4a4a4a"
SUPPORT_COLOURS = ("#2f3b4a", "#7f8c9b", "#d3d9df")   # >= 95, 80-94, < 80
FONT, FONT_SMALL, FONT_PILL = 7, 6, 5

# columns of the grid, left to right: (alignment, model in the table, header)
COLUMNS = [
    ("einsi",    "JTT+F+R10", "E-INS-i"),
    ("linsi",    "JTT+F+R10", "L-INS-i"),
    ("ginsi",    "JTT+F+R10", "G-INS-i"),
    ("mft",      "JTT+F+R10", "auto"),
    ("muscle",   "JTT+F+R11", "MUSCLE"),
    ("clustalo", "JTT+F+R10", "ClustalO"),
    ("einsi",    "LG+F+R10",  "E-INS-i"),
]

# tips of the collapsed tree: (label, number of sequences, colour, row)
TIPS = [
    ("Hagfish",        "43",  "hagfish",        0),
    ("Zebrafish",      "52",  "zebrafish",      1),
    ("Catsharks",      "3",   "catshark",       2.35),
    ("Zebrafish",      "5",   "zebrafish",      3.35),
    ("Human class I",  "62",  "human class I",  4.35),
    ("Zebrafish",      "46",  "zebrafish",      6.45),
    ("Catshark",       "1",   "catshark",       7.45),
    ("Human class II", "371", "human class II", 8.45),
]
ROW_CLASS_I, ROW_CLASS_II, ROW_BOTH = 3.35, 7.45, 5.4
# row of each clade of the table
CLADE_ROW = {
    "Hagfish": 0,
    "Zebrafish clade": 1,
    "Class I group": ROW_CLASS_I,
    "Class I + II groups": ROW_BOTH,
    "Class II group": ROW_CLASS_II,
}

FIGSIZE = (4, 3)                # in, as in scripts/plotting_functions.py
WIDTH, HEIGHT = 88.0, 42.0      # mm, the drawing
ROW_HEIGHT, FIRST_ROW_Y = 3.2, 13.2
X_ROOT, X_ZEBRAFISH_NODE, X_BOTH_NODE, X_CLASS_NODE = 0.9, 3.3, 5.7, 8.7
X_TIP_START, X_TIP_END, X_LABEL = 12.2, 15.6, 16.4
X_FIRST_COLUMN, COLUMN_STEP = 42.6, 5.75
PILL_WIDTH, PILL_HEIGHT = 5.35, 2.9


def row_y(row):
    return FIRST_ROW_Y + row * ROW_HEIGHT


def tint(hex_colour, fraction):
    r, g, b = [int(hex_colour[i:i + 2], 16) for i in (1, 3, 5)]
    return "#%02x%02x%02x" % tuple(int(255 - (255 - v) * fraction) for v in (r, g, b))


def support_colours(value):
    """Fill and text colour of one half of a pill."""
    if round(value) >= 95:
        return SUPPORT_COLOURS[0], "white"
    if round(value) >= 80:
        return SUPPORT_COLOURS[1], "white"
    return SUPPORT_COLOURS[2], INK


table = pd.read_csv("data/ReferenceTree/clade_support.csv")
support = {(row.clade, row.alignment, row.model): row for row in table.itertuples()}

fig = plt.figure(figsize=FIGSIZE)
ax = fig.add_axes([0, 0, 1, 1])
ax.axis("off")
# one axis unit is one millimetre; the margins centre the drawing in the figure
margin_x = (FIGSIZE[0] * 25.4 - WIDTH) / 2
margin_y = (FIGSIZE[1] * 25.4 - HEIGHT) / 2
ax.set_xlim(-margin_x, WIDTH + margin_x)
ax.set_ylim(HEIGHT + margin_y, -margin_y)

# x of each column; the LG column is set apart
column_x = [X_FIRST_COLUMN + j * COLUMN_STEP + (1.2 if model.startswith("LG") else 0)
            for j, (alignment, model, header) in enumerate(COLUMNS)]
x_end = column_x[-1] + 3.0

# shaded bands behind the class I and class II groups
for first_row, last_row, group in ((2.35, 4.35, "human class I"), (6.45, 8.45, "human class II")):
    ax.add_patch(FancyBboxPatch((7.7, row_y(first_row) - 1.7), x_end - 7.7, row_y(last_row) - row_y(first_row) + 3.4,
                                boxstyle="round,pad=0,rounding_size=0.8", fc=tint(COLOUR[group], 0.25),
                                ec="none", zorder=0))

fig.canvas.draw()
renderer = fig.canvas.get_renderer()


def write(x, y, parts):
    """Pieces of text (text, size, colour, weight) on one baseline, one after the other."""
    for text, size, colour, weight in parts:
        if not text:
            continue
        t = ax.text(x, y, text, va="center_baseline", ha="left", fontsize=size, color=colour, fontweight=weight)
        x += t.get_window_extent(renderer).width / fig.dpi * 25.4


def branch(x_node, y_node, x_child, y_child):
    ax.plot([x_node, x_node, x_child], [y_node, y_child, y_child], color=BRANCH, lw=0.6,
            solid_capstyle="butt", solid_joinstyle="miter", zorder=2)


# ---- collapsed tree ----
for label, n_sequences, group, row in TIPS:
    y = row_y(row)
    ax.add_patch(Polygon([[X_TIP_START, y], [X_TIP_END, y - 1.05], [X_TIP_END, y + 1.05]],
                         closed=True, fc=COLOUR[group], ec="none", zorder=3))
    if "class" in label:                      # the class name in bold
        plain, bold = label[:label.index(" class")], label[label.index(" class"):]
    else:
        plain, bold = label, ""
    write(X_LABEL, y, [(plain, FONT, INK, "normal"), (bold, FONT, INK, "bold"),
                       (" " + n_sequences, FONT_SMALL, MUTED, "normal")])

y_class_i, y_class_ii, y_both = row_y(ROW_CLASS_I), row_y(ROW_CLASS_II), row_y(ROW_BOTH)
y_zebrafish_node = (row_y(1) + y_both) / 2
y_root = (row_y(0) + y_zebrafish_node) / 2
for row in (2.35, 3.35, 4.35):
    branch(X_CLASS_NODE, y_class_i, X_TIP_START, row_y(row))
for row in (6.45, 7.45, 8.45):
    branch(X_CLASS_NODE, y_class_ii, X_TIP_START, row_y(row))
branch(X_BOTH_NODE, y_both, X_CLASS_NODE, y_class_i)
branch(X_BOTH_NODE, y_both, X_CLASS_NODE, y_class_ii)
branch(X_ZEBRAFISH_NODE, y_zebrafish_node, X_TIP_START, row_y(1))
branch(X_ZEBRAFISH_NODE, y_zebrafish_node, X_BOTH_NODE, y_both)
branch(X_ROOT, y_root, X_TIP_START, row_y(0))
branch(X_ROOT, y_root, X_ZEBRAFISH_NODE, y_zebrafish_node)
ax.plot([X_BOTH_NODE], [y_both], marker="o", ms=2.2, color=BRANCH, zorder=4)

# amphioxus, dashed: its position is not supported
y_amphioxus = row_y(0) - 2.7
ax.plot([6.8, 6.8, X_TIP_START], [row_y(0), y_amphioxus, y_amphioxus], color=COLOUR["amphioxus"], lw=0.8, ls=(0, (3, 2)))
write(X_TIP_START + 0.8, y_amphioxus, [("Amphioxus", FONT, INK, "normal"), (" 1, unstable", FONT_SMALL, MUTED, "normal")])

# ---- column headers and the model rules above them ----
for x, (alignment, model, header) in zip(column_x, COLUMNS):
    main_tree = (alignment, model) == ("einsi", "JTT+F+R10")
    ax.text(x - 0.9, FIRST_ROW_Y - 2.0, header, ha="left", va="bottom", rotation=35, rotation_mode="anchor",
            fontsize=FONT, color=INK, fontweight="bold" if main_tree else "normal")
y_rule = 3.0
ax.plot([column_x[0] + 0.3, column_x[5] + 7.0], [y_rule, y_rule], color=MUTED, lw=0.5)
ax.text((column_x[0] + column_x[5] + 7.3) / 2, y_rule - 0.5, "JTT", ha="center", va="bottom", fontsize=FONT, color=INK)
ax.plot([column_x[6] + 2.2, column_x[6] + 7.4], [y_rule, y_rule], color=MUTED, lw=0.5)
ax.text(column_x[6] + 4.8, y_rule - 0.5, "LG", ha="center", va="bottom", fontsize=FONT, color=INK)

# ---- grid of pills: left half SH-aLRT, right half bootstrap ----
for clade, row in CLADE_ROW.items():
    y = row_y(row)
    if clade == "Class I + II groups":
        ax.text(X_FIRST_COLUMN - 3.7, y, "I + II", ha="right", va="center_baseline", fontsize=FONT, color=INK, style="italic")
        ax.plot([X_BOTH_NODE + 0.7, X_FIRST_COLUMN - 10.6], [y, y], color="#b5b5b5", lw=0.4, ls=(0, (1, 2)), zorder=1)
    for x, (alignment, model, header) in zip(column_x, COLUMNS):
        values = support[clade, alignment, model]
        in_ml_tree = values.in_ml_tree == 1
        halves = [(-1, values.sh_alrt if in_ml_tree else None),
                  (+1, values.bootstrap_percent if in_ml_tree else None)]
        for side, value in halves:
            if value is None:
                # clade not in the ML tree: open pill, the bootstrap percentage is still shown
                fill, text_colour, edge = "white", MUTED, "#9a9a9a"
                text = "–" if side < 0 else f"{values.bootstrap_percent:.0f}"
            else:
                fill, text_colour = support_colours(value)
                edge = "none"
                text = f"{value:.0f}"
            pill = FancyBboxPatch((x - PILL_WIDTH / 2, y - PILL_HEIGHT / 2), PILL_WIDTH, PILL_HEIGHT,
                                  boxstyle=f"round,pad=0,rounding_size={PILL_HEIGHT / 2}",
                                  fc=fill, ec=edge, lw=0.4, zorder=3)
            ax.add_patch(pill)
            # show only this half of the pill
            clip_x = x - PILL_WIDTH / 2 - 0.3 if side < 0 else x + 0.12
            pill.set_clip_path(plt.Rectangle((clip_x, y - PILL_HEIGHT / 2 - 0.3), PILL_WIDTH / 2 + 0.18,
                                             PILL_HEIGHT + 0.6, transform=ax.transData))
            ax.text(x + side * (PILL_WIDTH / 4 + 0.05), y, text, ha="center", va="center_baseline",
                    fontsize=FONT_PILL - 0.8 if len(text) > 2 else FONT_PILL, color=text_colour, zorder=4)

# ---- legend, top left ----
ax.text(0.6, 2.0, "SH-aLRT | bootstrap (%)", ha="left", va="center_baseline", fontsize=FONT, color=INK)
x = 1.7
for fill, label, step in ((SUPPORT_COLOURS[0], "≥ 95", 8.2), (SUPPORT_COLOURS[1], "80–94", 9.6),
                          (SUPPORT_COLOURS[2], "< 80", 7.6)):
    ax.add_patch(Circle((x, 5.6), 1.05, fc=fill, ec="none"))
    ax.text(x + 1.6, 5.6, label, va="center_baseline", fontsize=FONT_SMALL, color=INK)
    x += step
ax.add_patch(Circle((x, 5.6), 0.95, fc="white", ec="#9a9a9a", lw=0.5))
ax.text(x + 1.6, 5.6, "not in ML tree", va="center_baseline", fontsize=FONT_SMALL, color=INK)

fig.savefig("figures/six_species_clade_support.svg")
plt.close(fig)
print("figures/six_species_clade_support.svg")
