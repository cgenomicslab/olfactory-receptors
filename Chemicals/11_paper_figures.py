"""Step 11 -- the assembled figures for the manuscript.

05 draws one figure per question, at whatever size suits that question. This script
composes the panels that go in the paper, at the page dimensions and type sizes Nature
Ecology and Evolution asks for, so nothing has to be rescaled afterwards (rescaling a
finished figure is what breaks type size).

  Figure 4    "Chemical space", double column, about two thirds of the page height
              a  odorants on the natural-product map
              b  what kind of organism odorants are isolated from
              c  biosynthetic pathway
              d  functional groups

  Supplementary
              a,b  the aromatic/aliphatic split of odour space
              c-f  receptor class on the map, and volatility against all of COCONUT
              g,h  source organisms at order and family level

Sizes follow the journal: 180 mm for a double-column figure, type between 5 and 7 pt at
final size, panel letters bold. Everything is drawn at final size, never scaled.

Outputs
-------
figures/Figure4_chemical_space.svg|pdf
figures/SupplementaryFigure_chemical_space.svg|pdf
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal, mannwhitneyu, rankdata

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

# Page geometry, in millimetres, converted once. A double-column figure is 180 mm wide.
MM = 1 / 25.4
DOUBLE_COLUMN = 180 * MM

ODORANT_COLOUR = "#0b7350"      # the odorant scatter in figure 4a, and the supplementary map
# The bar charts in figure 4b-d get their own colour rather than reusing the green above.
# Panel a is the only place green means "an odorant point on the map"; using the same green
# for bars in the neighbouring panels made the two read as the same encoding when they are
# not. Keep these two distinct.
ODORANT_BAR_COLOUR = "#48546b"
BACKGROUND_COLOUR = "#bfae94"
AROMATIC_COLOUR = "#8a6a3b"
M2OR_COLOUR = "#5b2c83"
FAINT_COLOUR = "#e9e5db"
CLASS_I_COLOUR, CLASS_II_COLOUR, SHARED_COLOUR = "#7f1734", "#557c99", "#9a8f80"
INK, MUTED, GRID = "#3f3f3c", "#8a8a86", "#e8e8e4"

# Type sizes for print. The journal floor is 5 pt; 6-7 pt is comfortable and still fits.
mpl.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 600,
    "svg.fonttype": "none", "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.edgecolor": "#bbbbb8", "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
})

PANEL_LETTER = {"fontsize": 9, "fontweight": "bold", "color": INK,
                "va": "top", "ha": "left"}

KINGDOM_ORDER = ["Viridiplantae", "Metazoa", "Fungi", "Bacteria"]
KINGDOM_LABEL = {"Viridiplantae": "plants", "Metazoa": "animals",
                 "Fungi": "fungi", "Bacteria": "bacteria"}

TAXON_EXAMPLES = {
    "Vitales": "grape", "Brassicales": "mustard", "Myrtales": "eucalyptus",
    "Zingiberales": "ginger", "Apiales": "carrot, cumin", "Poales": "grasses",
    "Lamiales": "mint, basil", "Sapindales": "citrus", "Rosales": "rose, hops",
    "Solanales": "tomato", "Asparagales": "onion, vanilla",
    "Xylariales": "fungi", "Pleosporales": "fungi", "Oscillatoriales": "cyanobacteria",
    "Haplosclerida": "sponges", "Dictyoceratida": "sponges",
    "Malacalcyonacea": "soft corals",
    "Caricaceae": "papaya", "Vitaceae": "grape", "Anacardiaceae": "mango, cashew",
    "Myrtaceae": "eucalyptus", "Cannabaceae": "hops", "Zingiberaceae": "ginger",
    "Amaryllidaceae": "onion, garlic", "Meliaceae": "neem",
    "Rhodomelaceae": "red algae", "Polyporaceae": "bracket fungi",
    "Celastraceae": "spindle trees", "Aspergillaceae": "moulds",
    "Alcyoniidae": "soft corals",
}

# Mammals appear as sources because a compound was measured in milk, breath or body
# odour rather than made there, so they are not a biosynthetic origin.
HOST_TAXA = {"Primates", "Artiodactyla", "Mammalia", "Chordata", "Hominidae",
             "Bovidae", "Carnivora", "Rodentia", "Aves", "Homo sapiens"}


def save(figure, name):
    """Write one figure as SVG and PDF, without a bounding-box refit.

    `bbox_inches="tight"` is deliberately not used: it changes the figure size, which
    would undo the point of laying the panels out at print dimensions.
    """
    FIGURES.mkdir(exist_ok=True)
    for extension in ("svg", "pdf"):
        figure.savefig(FIGURES / f"{name}.{extension}")
    plt.close(figure)
    size = figure.get_size_inches()
    print(f"wrote figures/{name}.svg|pdf   "
          f"({size[0] * 25.4:.0f} x {size[1] * 25.4:.0f} mm)", flush=True)


def centre_on_page(figure):
    """
    Shift every axis so the drawn content sits centred in the canvas.

    Margins set in the gridspec are measured to the axes, not to the ink, and tick
    labels and titles overhang by different amounts on each side -- so a layout with
    tidy-looking margins still lands off-centre on the page. This measures where the ink
    actually is and moves everything to balance it.

    Call last, once every panel, letter and caption is placed: those all count as ink,
    and centring before they exist leaves the figure balanced horizontally but sitting
    high or low on the page.
    """
    figure.canvas.draw()
    ink = figure.get_tightbbox(figure.canvas.get_renderer())
    width, height = figure.get_size_inches()

    shift_x = ((width - ink.width) / 2 - ink.x0) / width
    shift_y = ((height - ink.height) / 2 - ink.y0) / height

    for axis in figure.axes:
        box = axis.get_position()
        axis.set_position([box.x0 + shift_x, box.y0 + shift_y,
                           box.width, box.height])

    # Panel letters and captions live in figure coordinates, so they move too --
    # otherwise they would drift away from the panels they belong to.
    for text in figure.texts:
        x, y = text.get_position()
        text.set_position((x + shift_x, y + shift_y))
    return shift_x, shift_y


def letter(figure, text, x, y):
    """Place a bold panel letter at a fixed position on the page."""
    figure.text(x, y, text, **PANEL_LETTER)


def letter_for(figure, axis, text, pad_x=0.055, pad_y=0.022):
    """
    Put a panel letter above and left of an axis, measured from the axis box itself.

    Deriving the position from the axis rather than hard-coding it means the letters
    follow the layout when row heights or margins change, instead of drifting onto a
    neighbouring panel.
    """
    box = axis.get_position()
    figure.text(box.x0 - pad_x, box.y1 + pad_y, text, **PANEL_LETTER)


def load_universe():
    """The universe, ordered to match the embedding and the map coordinates."""
    universe = pd.read_parquet(RESULTS / "universe.parquet")
    index = pd.read_csv(RESULTS / "embed_index.csv")
    return universe.set_index("inchikey").loc[index.inchikey].reset_index()


def load_umap():
    """Map coordinates, preferring the version built on all 768 dimensions."""
    raw = RESULTS / "umap_raw768.npy"
    return np.load(raw if raw.exists() else RESULTS / "umap.npy")


def cliffs_delta(a, b, sample=100_000, seed=0):
    """How often a random member of `a` exceeds one of `b`, rescaled to -1..+1."""
    rng = np.random.default_rng(seed)
    if len(b) > sample:
        b = rng.choice(b, sample, replace=False)
    ranks = rankdata(np.concatenate([a, b]))
    u = ranks[:len(a)].sum() - len(a) * (len(a) + 1) / 2
    return float(2 * u / (len(a) * len(b)) - 1)


# --------------------------------------------------------------------- panels
def panel_map(axis, universe, coordinates, point=0.8, m2or_scale=1.4):
    """Odorants on the natural-product map, with the M2OR subset picked out.

    ``m2or_scale`` enlarges the 754 M2OR points relative to everything else so they stay
    findable among 720,445 background points. It is a compromise: too large and the M2OR
    layer hides the wider odorant set beneath it, which is the comparison the panel is
    there to make.
    """
    is_odorant = universe.is_odorant.values == 1
    in_m2or = (universe.in_m2or == 1).values

    background = np.flatnonzero(~is_odorant)
    others = np.flatnonzero(is_odorant & ~in_m2or)
    m2or = np.flatnonzero(in_m2or)

    axis.scatter(coordinates[background, 0], coordinates[background, 1],
                 s=point, c=BACKGROUND_COLOUR, alpha=0.22, lw=0, rasterized=True,
                 label=f"natural products ({len(background):,})")
    axis.scatter(coordinates[others, 0], coordinates[others, 1],
                 s=point, c=ODORANT_COLOUR, alpha=0.55, lw=0, rasterized=True,
                 label=f"odorants ({len(others):,})")
    axis.scatter(coordinates[m2or, 0], coordinates[m2or, 1],
                 s=point * m2or_scale, c=M2OR_COLOUR, alpha=0.8, lw=0, rasterized=True,
                 zorder=4, label=f"M2OR ({len(m2or):,})")

    # UMAP axes are arbitrary but isotropic: stretching one against the other would
    # distort apparent neighbourhoods, so the map is drawn square whatever the panel
    # aspect. This also frees the margins either side for the legend.
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(frameon=False, markerscale=4, loc="upper left",
                labelcolor=INK, handletextpad=0.3, borderpad=0.2,
                bbox_to_anchor=(-0.02, 1.06))


def panel_kingdom(axis):
    """Source-organism composition, odorants against non-odorant natural products."""
    ranks = pd.read_csv(RESULTS / "organism_rank_enrichment.csv")
    shares = pd.concat([
        ranks.query("rank == 'kingdom' and taxon in @KINGDOM_ORDER"),
        ranks.query("rank == 'superkingdom' and taxon == 'Bacteria'"),
    ]).set_index("taxon")

    present = [k for k in KINGDOM_ORDER if k in shares.index]
    positions = np.arange(len(present))[::-1]
    odorant = [100 * shares.mean_share_odorant[k] for k in present]
    other = [100 * shares.mean_share_background[k] for k in present]

    axis.barh(positions + 0.19, odorant, height=0.36, color=ODORANT_BAR_COLOUR,
              label="odorants")
    axis.barh(positions - 0.19, other, height=0.36, color=BACKGROUND_COLOUR,
              label="other natural products")
    for position, a, b, taxon in zip(positions, odorant, other, present):
        axis.text(max(a, b) + 2, position,
                  f"{shares.mean_share_odorant[taxon] / shares.mean_share_background[taxon]:.2f}×",
                  va="center", fontsize=6, color=MUTED)

    axis.set_yticks(positions)
    axis.set_yticklabels([KINGDOM_LABEL[k] for k in present])
    axis.set_xlim(0, 118)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("share of a molecule's source organisms (%)")
    axis.legend(frameon=False, loc="lower right", labelcolor=INK,
                handlelength=1.1, handletextpad=0.4, borderpad=0.2)
    axis.tick_params(left=False)


def panel_pathway(axis):
    """Biosynthetic pathway, odorants against the background."""
    table = pd.read_csv(RESULTS / "pathway_enrichment.csv").sort_values("pct_odorant")
    positions = np.arange(len(table))

    axis.barh(positions + 0.19, table.pct_odorant, height=0.36, color=ODORANT_BAR_COLOUR,
              label="odorants")
    axis.barh(positions - 0.19, table.pct_background, height=0.36,
              color=BACKGROUND_COLOUR, label="natural products")

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [p.replace(" and ", "\n") for p in table.pathway], fontsize=6)
    axis.set_xlabel("% of annotated molecules")
    axis.legend(frameon=False, loc="lower right", labelcolor=INK,
                handlelength=1.1, handletextpad=0.4, borderpad=0.2)
    axis.tick_params(left=False)


def panel_functional_groups(axis):
    """Functional groups as log odds, odorants against the background."""
    table = pd.read_csv(RESULTS / "fg_enrichment.csv").sort_values("odds_ratio")
    positions = np.arange(len(table))
    log_odds = np.log2(table.odds_ratio.clip(lower=1e-3))
    colours = [ODORANT_BAR_COLOUR if v > 0 else BACKGROUND_COLOUR for v in log_odds]

    axis.barh(positions, log_odds, color=colours, height=0.68)
    axis.axvline(0, color=GRID, lw=0.8)
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [g.replace("fg_", "").replace("_", " ") for g in table.group], fontsize=6)
    axis.set_xlabel("log$_2$ odds ratio (odorant vs natural product)")
    axis.tick_params(left=False)


# ------------------------------------------------------------------ figure 4
def figure4(height_mm=118):
    """The main chemical-space figure.

    ``height_mm`` sets the drawn height only; type sizes come from the rcParams above and
    are never scaled with it, so compacting the figure shrinks the plotting area and the
    whitespace around it while leaving every label at its print size. The default of
    118 mm is close to half the text height of a page. The second row is given the larger
    share because panels c and d are categorical bar charts whose height is set by their
    row count (7 pathways and 16 functional groups), not by the data range.
    """
    universe = load_universe()
    coordinates = load_umap()

    height = height_mm * MM
    figure = plt.figure(figsize=(DOUBLE_COLUMN, height))
    grid = figure.add_gridspec(
        2, 2, height_ratios=[0.80, 1.20], width_ratios=[1, 1],
        left=0.135, right=0.99, top=0.945, bottom=0.095, hspace=0.34, wspace=0.42)

    panel_map(figure.add_subplot(grid[0, 0]), universe, coordinates)
    panel_kingdom(figure.add_subplot(grid[0, 1]))
    panel_pathway(figure.add_subplot(grid[1, 0]))
    panel_functional_groups(figure.add_subplot(grid[1, 1]))

    for axis, text, pad in zip(figure.axes, "abcd", [0.055, 0.075, 0.105, 0.090]):
        letter_for(figure, axis, text, pad_x=pad, pad_y=0.020)

    centre_on_page(figure)
    save(figure, "Figure4_chemical_space")


# ------------------------------------------------------- supplementary panels
def panel_aromatic_map(axis, universe, coordinates):
    """Odorants split by whether they carry a benzene ring."""
    is_odorant = universe.is_odorant.values == 1
    aromatic_ring = universe["AromaticRings"].values > 0
    aliphatic = is_odorant & ~aromatic_ring
    aromatic = is_odorant & aromatic_ring
    percent = 100 * aliphatic.sum() / is_odorant.sum()

    background = np.flatnonzero(~is_odorant)
    axis.scatter(coordinates[background, 0], coordinates[background, 1],
                 s=0.7, c=FAINT_COLOUR, alpha=0.3, lw=0, rasterized=True)
    axis.scatter(coordinates[aliphatic, 0], coordinates[aliphatic, 1], s=1.4,
                 c=ODORANT_COLOUR, alpha=0.7, lw=0, rasterized=True,
                 label=f"aliphatic ({percent:.0f}%)")
    axis.scatter(coordinates[aromatic, 0], coordinates[aromatic, 1], s=1.4,
                 c=AROMATIC_COLOUR, alpha=0.7, lw=0, rasterized=True,
                 label=f"aromatic ({100 - percent:.0f}%)")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.legend(frameon=False, markerscale=4, loc="upper right", labelcolor=INK,
                handletextpad=0.3, borderpad=0.2)


def panel_aromatic_difference(axis, universe):
    """How the aromatic and aliphatic halves differ, in standardised differences."""
    is_odorant = universe.is_odorant.values == 1
    aromatic_ring = universe["AromaticRings"].values > 0
    aliphatic = is_odorant & ~aromatic_ring
    aromatic = is_odorant & aromatic_ring

    # FracCsp3 is left out on purpose: aromatic carbons are sp2, so the difference is the
    # definition of aromatic rather than a finding, and its size hides everything else.
    wanted = ["Tb_joback", "HBA", "TPSA", "MolWt", "nO", "LogP", "RotBonds"]
    names, values = [], []
    for name in wanted:
        if name not in universe:
            continue
        column = pd.to_numeric(universe[name], errors="coerce").values.astype(float)
        first = column[aromatic][np.isfinite(column[aromatic])]
        second = column[aliphatic][np.isfinite(column[aliphatic])]
        spread = np.sqrt((np.nanvar(first) + np.nanvar(second)) / 2)
        names.append(name)
        values.append((np.nanmean(first) - np.nanmean(second)) / spread if spread else 0)

    order = np.argsort(values)
    names = [names[i] for i in order]
    values = [values[i] for i in order]
    positions = np.arange(len(names))

    axis.barh(positions, values, height=0.62,
              color=[AROMATIC_COLOUR if v > 0 else ODORANT_COLOUR for v in values])
    axis.axvline(0, color=GRID, lw=0.8)
    for position, value in zip(positions, values):
        axis.text(value + (0.02 if value >= 0 else -0.02), position, f"{value:+.2f}",
                  va="center", ha="left" if value >= 0 else "right",
                  fontsize=6, color=MUTED)
    axis.set_yticks(positions)
    axis.set_yticklabels(names, fontsize=6.5)
    limit = max(abs(min(values)), abs(max(values))) * 1.5
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("standardised difference\n(← aliphatic higher | aromatic higher →)")
    axis.tick_params(left=False)


def receptor_class_groups(universe, coordinates):
    """Panel molecules with a receptor-class assignment, and their map positions."""
    join = pd.read_csv(HERE / "panel_universe_join.csv")
    join = join[join.in_universe].copy()

    positions = universe.reset_index(drop=True)
    positions["row"] = np.arange(len(positions))
    join = join.merge(positions[["inchikey", "row", "Tb_joback"]], on="inchikey",
                      how="inner", suffixes=("", "_uni"))
    join["x"] = coordinates[join.row.values, 0]
    join["y"] = coordinates[join.row.values, 1]
    return join


def panel_class_map(axis, coordinates, join, group, colour):
    """One receptor-class group on the map."""
    subset = join[join.class_group == group]
    sample = np.random.default_rng(0).choice(len(coordinates),
                                             min(120_000, len(coordinates)),
                                             replace=False)
    axis.scatter(coordinates[sample, 0], coordinates[sample, 1], s=0.5,
                 c=FAINT_COLOUR, alpha=0.35, lw=0, rasterized=True)
    axis.scatter(subset.x, subset.y, s=3.0, c=colour, alpha=0.85, lw=0, rasterized=True)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_title(f"{group}  (n={len(subset)})", color=colour, fontsize=6.5, pad=2)


def panel_class_volatility(axis, universe, join):
    """
    Volatility by receptor class, against every other natural product.

    The three receptor-class groups are indistinguishable from each other. Including the
    rest of COCONUT is what makes that null readable: the same axis carries a difference
    of Cliff's delta about -0.9, so the comparison plainly has the power to see one.
    """
    groups, labels, colours = [], [], []
    for group, colour in [("Class I only", CLASS_I_COLOUR), ("shared", SHARED_COLOUR),
                          ("Class II only", CLASS_II_COLOUR)]:
        values = pd.to_numeric(join.Tb_joback[join.class_group == group],
                               errors="coerce").dropna().values
        groups.append(values)
        labels.append({"Class I only": "class I", "shared": "shared",
                       "Class II only": "class II"}[group])
        colours.append(colour)

    rest = pd.to_numeric(universe.Tb_joback[universe.in_background == 1],
                         errors="coerce").dropna().values
    groups.append(rest)
    labels.append("other\nNPs")
    colours.append(BACKGROUND_COLOUR)

    for position, (values, colour) in enumerate(zip(groups, colours)):
        # The background has 648,000 members; draw a sample so the strip stays readable.
        shown = values
        if len(shown) > 800:
            shown = np.random.default_rng(0).choice(shown, 800, replace=False)
        jitter = position + np.random.default_rng(position).uniform(-0.17, 0.17, len(shown))
        axis.scatter(jitter, shown, s=1.6, color=colour, alpha=0.45, lw=0, rasterized=True)
        low, middle, high = np.percentile(values, [25, 50, 75])
        axis.plot([position - 0.32, position + 0.32], [middle] * 2, color=INK, lw=1.6,
                  solid_capstyle="round", zorder=5)
        axis.plot([position, position], [low, high], color=INK, lw=0.8, zorder=4)

    statistic, p_value = kruskal(*groups[:3])
    delta = cliffs_delta(groups[0], rest)
    axis.set_xticks(range(4))
    # Four labels do not fit side by side in a quarter-width column, so they lean.
    axis.set_xticklabels(labels, fontsize=6, rotation=45, ha="right",
                         rotation_mode="anchor")
    axis.set_ylabel("boiling point (K)")
    axis.set_ylim(250, 1750)
    axis.set_title(f"classes alike, $P$={p_value:.2f}\n"
                   f"all $\\ll$ other NPs, $\\delta$={delta:.2f}",
                   fontsize=6, color=INK, pad=3)
    axis.tick_params(axis="x", length=0)


def panel_taxon_rank(axis, rank, n_shown=10):
    """Enriched and depleted taxa at one rank, as log fold change."""
    table = pd.read_csv(RESULTS / "organism_rank_enrichment.csv")
    at_rank = table[(table["rank"] == rank) & (table.q < 0.05)].dropna(
        subset=["share_ratio"])
    at_rank = at_rank[~at_rank.taxon.isin(HOST_TAXA)]
    substantial = at_rank[
        at_rank[["mean_share_odorant", "mean_share_background"]].max(axis=1) >= 0.005
    ].sort_values("share_ratio", ascending=False)

    half = n_shown // 2
    shown = pd.concat([substantial.head(half), substantial.tail(half)])
    positions = np.arange(len(shown))[::-1]
    log_ratio = np.log2(shown.share_ratio.values)

    axis.barh(positions, log_ratio, height=0.66,
              color=[ODORANT_COLOUR if v > 0 else BACKGROUND_COLOUR for v in log_ratio])
    axis.axvline(0, color=GRID, lw=0.8)
    for position, value, ratio in zip(positions, log_ratio, shown.share_ratio):
        axis.text(value + (0.12 if value >= 0 else -0.12), position, f"{ratio:.1f}×",
                  va="center", ha="left" if value >= 0 else "right",
                  fontsize=5.5, color=MUTED)

    axis.set_yticks(positions)
    axis.set_yticklabels(
        [f"{t}\n{TAXON_EXAMPLES[t]}" if t in TAXON_EXAMPLES else t
         for t in shown.taxon], fontsize=5.5)
    limit = max(abs(log_ratio.min()), abs(log_ratio.max())) * 1.75
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("log$_2$ fold change in source share")
    axis.set_title(rank.capitalize(), fontsize=7, color=INK, pad=3)
    axis.tick_params(left=False)


def supplementary():
    """The supplementary chemical-space figure."""
    universe = load_universe()
    coordinates = load_umap()
    join = receptor_class_groups(universe, coordinates)

    height = 225 * MM
    figure = plt.figure(figsize=(DOUBLE_COLUMN, height))
    grid = figure.add_gridspec(
        3, 4, height_ratios=[1.0, 0.80, 1.20],
        left=0.135, right=0.985, top=0.945, bottom=0.075, hspace=0.48, wspace=0.75)

    # Row 1 -- the aromatic split.
    axis_a = figure.add_subplot(grid[0, :2])
    panel_aromatic_map(axis_a, universe, coordinates)
    axis_b = figure.add_subplot(grid[0, 2:])
    panel_aromatic_difference(axis_b, universe)

    # Row 2 -- receptor class on the map, then volatility against all of COCONUT.
    class_axes = []
    for column, (group, colour) in enumerate([
            ("Class I only", CLASS_I_COLOUR), ("shared", SHARED_COLOUR),
            ("Class II only", CLASS_II_COLOUR)]):
        axis = figure.add_subplot(grid[1, column])
        panel_class_map(axis, coordinates, join, group, colour)
        class_axes.append(axis)
    axis_f = figure.add_subplot(grid[1, 3])
    panel_class_volatility(axis_f, universe, join)

    # Row 3 -- source organisms at finer ranks.
    axis_g = figure.add_subplot(grid[2, :2])
    panel_taxon_rank(axis_g, "order")
    axis_h = figure.add_subplot(grid[2, 2:])
    panel_taxon_rank(axis_h, "family")

    letter_for(figure, axis_a, "a", pad_x=0.055)
    letter_for(figure, axis_b, "b", pad_x=0.075)
    for axis, text in zip(class_axes, "cde"):
        letter_for(figure, axis, text, pad_x=0.028, pad_y=0.030)
    letter_for(figure, axis_f, "f", pad_x=0.045, pad_y=0.030)
    letter_for(figure, axis_g, "g", pad_x=0.100)
    letter_for(figure, axis_h, "h", pad_x=0.100)

    figure.text(0.5, 0.012,
                "Mammalian taxa are excluded from g and h: they appear as sources because "
                "a compound was measured in milk, breath or body odour, not synthesised "
                "there.", ha="center", fontsize=5.8, color=MUTED, style="italic")

    centre_on_page(figure)
    save(figure, "SupplementaryFigure_chemical_space")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="4s", help="4 for the main figure, s for supp")
    parser.add_argument("--height", type=float, default=118.0,
                        help="drawn height of figure 4 in mm; type size is unaffected")
    args = parser.parse_args()

    if "4" in args.only:
        figure4(height_mm=args.height)
    if "s" in args.only:
        supplementary()


if __name__ == "__main__":
    main()
