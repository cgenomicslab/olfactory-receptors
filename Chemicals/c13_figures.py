"""Draw every figure, one SVG per plot, in the shared style of scripts/plotting_functions.py.

    python c13_figures.py                  # everything
    python c13_figures.py --only paper     # or --only working

Figure 4:       odorant_map, organism_kingdom, biosynthetic_pathway,
                functional_groups_matched
Supplementary:  aromatic_map, aromatic_vs_aliphatic, receptor_class_map_class_I,
                receptor_class_map_shared, receptor_class_map_class_II,
                receptor_class_volatility, organism_order, organism_family
Working:        map_boiling_point, map_molecular_weight, map_logP,
                organism_kingdom_by_list, receptor_clusters_on_map,
                receptor_clusters_properties
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import kruskal

import chemspace as cs

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

sys.path.append(str(HERE.parent / "scripts"))
from plotting_functions import FIGSIZE, GREY, custom_plots  # noqa: E402

plt.rcParams.update(custom_plots())

ODORANT_COLOUR = "#0b7350"      # an odorant point on a map
# Bars get their own colour rather than reusing the green above: green means "an odorant
# point on the map", and bars in the same colour read as the same encoding.
ODORANT_BAR_COLOUR = "#48546b"
BACKGROUND_COLOUR = "#bfae94"
AROMATIC_COLOUR = "#8a6a3b"
M2OR_COLOUR = "#5b2c83"
FAINT_COLOUR = "#e9e5db"
CLASS_I_COLOUR, CLASS_II_COLOUR, SHARED_COLOUR = "#7f1734", "#557c99", "#9a8f80"
ZERO_LINE = "#e8e8e4"
NO_ESTIMATE_COLOUR = "#e8b21f"  # yellow shares no hue with the green ramp
OUTLINE_COLOUR = "#3f3f3c"
PROPERTY_RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "palm", ["#f1f6f3", "#9ac6b2", "#2f9068", "#0b7350", "#04301f"])
# Six clusters cannot be told apart by hue as overlaid points, so the cluster figure
# facets them; these colours only mark which facet a strip belongs to.
CLUSTER_COLOURS = ["#543005", "#ae7121", "#c8923b", "#98d7cd", "#24877f", "#003c30"]

# Numbers printed beside bars, a step smaller than the tick labels.
ANNOTATION = {"fontsize": 8, "color": GREY}

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
    "Malacalcyonacea": "soft corals", "Scleralcyonacea": "soft corals",
    "Aplousobranchia": "sea squirts",
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

CLASS_GROUPS = [("Class I only", "class_I", CLASS_I_COLOUR),
                ("shared", "shared", SHARED_COLOUR),
                ("Class II only", "class_II", CLASS_II_COLOUR)]


def save(figure, name):
    """Write one figure as SVG and close it."""
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote figures/{name}.svg", flush=True)


def load_universe():
    """The universe, ordered to match the embedding and the map coordinates."""
    universe = pd.read_parquet(RESULTS / "universe.parquet")
    index = pd.read_csv(RESULTS / "embed_index.csv")
    return universe.set_index("inchikey").loc[index.inchikey].reset_index()


def load_umap():
    """Map coordinates from c04, rows in the order of the universe above."""
    return np.load(RESULTS / "umap.npy")


def blank_map_axes(axis):
    """UMAP axes carry no units, so no ticks; equal aspect so neighbourhoods keep shape."""
    axis.set_aspect("equal", adjustable="datalim")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")


def paired_bars(axis, labels, odorant, other, odorant_label, other_label):
    """Two horizontal bars per category, odorants above the comparison group."""
    positions = np.arange(len(labels))[::-1]
    axis.barh(positions + 0.19, odorant, height=0.36, color=ODORANT_BAR_COLOUR,
              label=odorant_label)
    axis.barh(positions - 0.19, other, height=0.36, color=BACKGROUND_COLOUR,
              label=other_label)
    axis.set_yticks(positions)
    axis.set_yticklabels(labels)
    axis.tick_params(left=False)
    return positions


# ------------------------------------------------------------------- Figure 4
def odorant_map(universe, coordinates):
    """Odorants on the natural-product map, with the M2OR subset picked out.

    The 754 M2OR points are drawn slightly larger than the rest so they stay findable
    among 720,445 background points, but not so large that they hide the wider odorant
    set beneath them, which is the comparison the plot is there to make.
    """
    is_odorant = universe.is_odorant.values == 1
    in_m2or = (universe.in_m2or == 1).values
    background = np.flatnonzero(~is_odorant)
    others = np.flatnonzero(is_odorant & ~in_m2or)
    m2or = np.flatnonzero(in_m2or)

    figure, axis = plt.subplots(figsize=FIGSIZE)
    axis.scatter(coordinates[background, 0], coordinates[background, 1], s=0.5,
                 c=BACKGROUND_COLOUR, alpha=0.3, lw=0, rasterized=True,
                 label=f"natural products ({len(background):,})")
    axis.scatter(coordinates[others, 0], coordinates[others, 1], s=0.3,
                 c=ODORANT_COLOUR, alpha=0.55, lw=0, rasterized=True,
                 label=f"odorants ({len(others):,})")
    axis.scatter(coordinates[m2or, 0], coordinates[m2or, 1], s=0.6, c=M2OR_COLOUR,
                 alpha=0.8, lw=0, rasterized=True, zorder=4,
                 label=f"M2OR ({len(m2or):,})")
    blank_map_axes(axis)
    axis.legend(frameon=False, markerscale=8, loc="upper left",
                bbox_to_anchor=(1.0, 1.0))
    save(figure, "odorant_map")


def organism_kingdom():
    """Source-organism composition by kingdom, odorants against other natural products.

    Mean share of a molecule's source organisms in each group (c07). Bacteria are the
    superkingdom, not one of NCBI's two kingdom-rank halves of it.
    """
    ranks = pd.read_csv(RESULTS / "organism_rank_enrichment.csv")
    shares = pd.concat([
        ranks.query("rank == 'kingdom' and taxon in @KINGDOM_ORDER"),
        ranks.query("rank == 'superkingdom' and taxon == 'Bacteria'"),
    ]).set_index("taxon").loc[KINGDOM_ORDER]

    figure, axis = plt.subplots(figsize=FIGSIZE)
    odorant = 100 * shares.mean_share_odorant.values
    other = 100 * shares.mean_share_background.values
    positions = paired_bars(axis, [KINGDOM_LABEL[k] for k in KINGDOM_ORDER], odorant,
                            other, "odorants", "other natural products")
    for position, a, b, ratio in zip(positions, odorant, other, shares.share_ratio):
        axis.text(max(a, b) + 2, position, f"{ratio:.2f}×", va="center", **ANNOTATION)
    axis.set_xlim(0, 115)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("share of a molecule's source organisms (%)")
    axis.legend(frameon=False, loc="lower right")
    save(figure, "organism_kingdom")


def biosynthetic_pathway():
    """NPClassifier pathway, as a percentage of the annotated molecules on each side."""
    table = pd.read_csv(RESULTS / "pathway_enrichment.csv").sort_values(
        "pct_odorant", ascending=False)
    figure, axis = plt.subplots(figsize=(FIGSIZE[0], 1.5 * FIGSIZE[1]))
    paired_bars(axis, [p.replace(" and ", "\n") for p in table.pathway],
                table.pct_odorant.values, table.pct_background.values,
                "odorants", "natural products")
    axis.set_xlabel("% of annotated molecules")
    axis.legend(frameon=False, loc="lower right")
    save(figure, "biosynthetic_pathway")


def functional_groups_matched():
    """
    Functional groups as log odds, odorants against their matched natural products.

    The matched contrast (c05): each odorant against a natural product of the same size
    and volatility, so a group is not called depleted merely for sitting on large
    molecules. Whiskers are the 95% interval from resampling pairs; groups whose
    difference is not significant (McNemar, BH q >= 0.05) are drawn pale.
    """
    table = pd.read_csv(RESULTS / "fg_enrichment.csv").sort_values("OR_matched")
    positions = np.arange(len(table))
    log_odds = np.log2(table.OR_matched.values)
    low = np.log2(table.ci_lo_matched.values)
    high = np.log2(table.ci_hi_matched.values)
    significant = table.q_matched.values < 0.05

    figure, axis = plt.subplots(figsize=(FIGSIZE[0], 1.5 * FIGSIZE[1]))
    colours = [mpl.colors.to_rgba(ODORANT_BAR_COLOUR if value > 0 else BACKGROUND_COLOUR,
                                  1.0 if is_significant else 0.35)
               for value, is_significant in zip(log_odds, significant)]
    axis.barh(positions, log_odds, color=colours, height=0.68)
    axis.errorbar(log_odds, positions, xerr=[log_odds - low, high - log_odds],
                  fmt="none", ecolor=GREY, elinewidth=0.6, capsize=0)
    for position, value, top, bottom, ratio in zip(positions, log_odds, high, low,
                                                   table.OR_matched):
        edge = top if value >= 0 else bottom
        axis.text(edge + (0.15 if value >= 0 else -0.15), position, f"{ratio:.2f}",
                  va="center", ha="left" if value >= 0 else "right", **ANNOTATION)
    axis.axvline(0, color=ZERO_LINE, lw=0.8)
    # Room either side of the widest interval for its number label.
    axis.set_xlim(np.nanmin(low) - 1.3, np.nanmax(high) + 1.3)
    axis.set_yticks(positions)
    axis.set_yticklabels([g.replace("fg_", "").replace("_", " ") for g in table.group])
    axis.set_xlabel("log$_2$ odds ratio\n(odorant vs matched natural product)")
    axis.tick_params(left=False)
    save(figure, "functional_groups_matched")


def figure4():
    """The four plots of Figure 4, each its own file."""
    universe = load_universe()
    odorant_map(universe, load_umap())
    organism_kingdom()
    biosynthetic_pathway()
    functional_groups_matched()


# --------------------------------------------------------------- Supplementary
def aromatic_split(universe):
    """Odorants with and without an aromatic ring."""
    is_odorant = universe.is_odorant.values == 1
    aromatic_ring = universe["AromaticRings"].values > 0
    return is_odorant & ~aromatic_ring, is_odorant & aromatic_ring


def aromatic_map(universe, coordinates):
    """The aromatic and aliphatic odorants on the map."""
    aliphatic, aromatic = aromatic_split(universe)
    percent = 100 * aliphatic.sum() / (aliphatic.sum() + aromatic.sum())
    background = np.flatnonzero(universe.is_odorant.values != 1)

    figure, axis = plt.subplots(figsize=FIGSIZE)
    axis.scatter(coordinates[background, 0], coordinates[background, 1], s=0.3,
                 c=FAINT_COLOUR, alpha=0.3, lw=0, rasterized=True)
    axis.scatter(coordinates[aliphatic, 0], coordinates[aliphatic, 1], s=0.6,
                 c=ODORANT_COLOUR, alpha=0.7, lw=0, rasterized=True,
                 label=f"aliphatic ({percent:.0f}%)")
    axis.scatter(coordinates[aromatic, 0], coordinates[aromatic, 1], s=0.6,
                 c=AROMATIC_COLOUR, alpha=0.7, lw=0, rasterized=True,
                 label=f"aromatic ({100 - percent:.0f}%)")
    blank_map_axes(axis)
    axis.legend(frameon=False, markerscale=8, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    save(figure, "aromatic_map")


def aromatic_vs_aliphatic(universe):
    """How the aromatic and aliphatic odorants differ, in standardised differences.

    FracCsp3 is left out on purpose: aromatic carbons are sp2, so the difference is the
    definition of aromatic rather than a finding, and its size hides everything else.
    """
    aliphatic, aromatic = aromatic_split(universe)
    wanted = ["Tb_joback", "HBA", "TPSA", "MolWt", "nO", "LogP", "RotBonds"]
    values = []
    for name in wanted:
        column = pd.to_numeric(universe[name], errors="coerce").values.astype(float)
        values.append(cs.smd(column[aromatic], column[aliphatic]))

    order = np.argsort(values)
    names = [wanted[i] for i in order]
    values = [values[i] for i in order]
    positions = np.arange(len(names))

    figure, axis = plt.subplots(figsize=FIGSIZE)
    axis.barh(positions, values, height=0.62,
              color=[AROMATIC_COLOUR if v > 0 else ODORANT_COLOUR for v in values])
    axis.axvline(0, color=ZERO_LINE, lw=0.8)
    for position, value in zip(positions, values):
        axis.text(value + (0.02 if value >= 0 else -0.02), position, f"{value:+.2f}",
                  va="center", ha="left" if value >= 0 else "right", **ANNOTATION)
    axis.set_yticks(positions)
    axis.set_yticklabels(names)
    limit = max(abs(min(values)), abs(max(values))) * 1.5
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("standardised difference\n(← aliphatic higher | aromatic higher →)")
    axis.tick_params(left=False)
    save(figure, "aromatic_vs_aliphatic")


def receptor_panel(universe, coordinates):
    """The panel molecules c11 placed, with their groups, map position and properties."""
    join = pd.read_csv(RESULTS / "panel_universe_join.csv")
    join = join[join.in_universe].copy()
    rows = join.row.astype(int).values
    join["x"] = coordinates[rows, 0]
    join["y"] = coordinates[rows, 1]
    join["Tb_joback"] = pd.to_numeric(universe.Tb_joback.values[rows], errors="coerce")
    join["np_likeness"] = pd.to_numeric(universe.np_likeness.values[rows], errors="coerce")
    return join


def map_background(axis, coordinates):
    """A fixed random 120,000 of the universe in faint grey, for maps with few points."""
    sample = np.random.default_rng(0).choice(len(coordinates),
                                             min(120_000, len(coordinates)), replace=False)
    axis.scatter(coordinates[sample, 0], coordinates[sample, 1], s=0.3, c=FAINT_COLOUR,
                 alpha=0.4, lw=0, rasterized=True)


def receptor_class_maps(join, coordinates):
    """Each receptor-class group on the map, one file per group."""
    for group, stem, colour in CLASS_GROUPS:
        subset = join[join.class_group == group]
        figure, axis = plt.subplots(figsize=FIGSIZE)
        map_background(axis, coordinates)
        axis.scatter(subset.x, subset.y, s=4, c=colour, alpha=0.85, lw=0, rasterized=True)
        blank_map_axes(axis)
        axis.set_title(f"{group}  (n={len(subset)})", color=colour)
        save(figure, f"receptor_class_map_{stem}")


def receptor_class_volatility(universe, join):
    """
    Boiling point by receptor class, against every other natural product.

    The three groups are indistinguishable from each other. Including the rest of
    COCONUT is what makes that null readable: the same axis carries a difference of
    Cliff's delta about -0.9, so the comparison plainly has the power to see one.
    """
    groups, labels, colours = [], [], []
    for group, _, colour in CLASS_GROUPS:
        groups.append(join.Tb_joback[join.class_group == group].dropna().values)
        labels.append(group.replace(" only", ""))
        colours.append(colour)
    rest = pd.to_numeric(universe.Tb_joback[universe.in_background == 1],
                         errors="coerce").dropna().values
    groups.append(rest)
    labels.append("other NPs")
    colours.append(BACKGROUND_COLOUR)

    figure, axis = plt.subplots(figsize=FIGSIZE)
    for position, (values, colour) in enumerate(zip(groups, colours)):
        # The background has 648,000 members; a sample keeps the strip readable.
        shown = values
        if len(shown) > 800:
            shown = np.random.default_rng(0).choice(shown, 800, replace=False)
        jitter = position + np.random.default_rng(position).uniform(-0.17, 0.17, len(shown))
        axis.scatter(jitter, shown, s=3, color=colour, alpha=0.45, lw=0, rasterized=True)
        low, middle, high = np.percentile(values, [25, 50, 75])
        axis.plot([position - 0.32, position + 0.32], [middle] * 2, color=GREY, lw=1.6,
                  solid_capstyle="round", zorder=5)
        axis.plot([position, position], [low, high], color=GREY, lw=0.8, zorder=4)

    _, p_value = kruskal(*groups[:3])
    delta = cs.cliffs_delta(groups[0], rest, sample=100_000)
    axis.set_xticks(range(4))
    axis.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    axis.set_ylabel("boiling point (K)")
    axis.set_ylim(250, 1750)
    axis.set_title(f"classes alike, $P$ = {p_value:.2f}\n"
                   f"class I vs other NPs, $\\delta$ = {delta:.2f}", fontsize=9)
    axis.tick_params(axis="x", length=0)
    save(figure, "receptor_class_volatility")


def organism_rank(rank, n_shown=10):
    """Enriched and depleted taxa at one rank, as log fold change in source share.

    Taxa with q < 0.05 that account for at least 0.5% of sources on one side; below
    that the ratio of two near-zero shares is unstable. Host taxa are left out.
    """
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

    figure, axis = plt.subplots(figsize=(FIGSIZE[0], 1.5 * FIGSIZE[1]))
    axis.barh(positions, log_ratio, height=0.66,
              color=[ODORANT_COLOUR if v > 0 else BACKGROUND_COLOUR for v in log_ratio])
    axis.axvline(0, color=ZERO_LINE, lw=0.8)
    for position, value, ratio in zip(positions, log_ratio, shown.share_ratio):
        # One decimal hides a 25-fold depletion as "0.0x"; two significant digits do not.
        label = f"{ratio:.1f}×" if ratio >= 1 else f"{ratio:.2g}×"
        axis.text(value + (0.15 if value >= 0 else -0.15), position, label,
                  va="center", ha="left" if value >= 0 else "right", **ANNOTATION)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{t}\n{TAXON_EXAMPLES[t]}" if t in TAXON_EXAMPLES else t
                          for t in shown.taxon], fontsize=8)
    limit = max(abs(log_ratio.min()), abs(log_ratio.max())) * 1.6
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("log$_2$ fold change in source share")
    axis.set_title(rank.capitalize())
    axis.tick_params(left=False)
    save(figure, f"organism_{rank}")


def supplementary():
    """The plots of the Supplementary figure, each its own file."""
    universe = load_universe()
    coordinates = load_umap()
    join = receptor_panel(universe, coordinates)
    aromatic_map(universe, coordinates)
    aromatic_vs_aliphatic(universe)
    receptor_class_maps(join, coordinates)
    receptor_class_volatility(universe, join)
    organism_rank("order")
    organism_rank("family")


# ------------------------------------------------------------ working figures
PROPERTIES = [
    ("Tb_joback", "boiling point (K)", "map_boiling_point"),
    ("MolWt", "molecular weight (Da)", "map_molecular_weight"),
    ("LogP", "logP", "map_logP"),
]


def density_contour(axis, x, y, extent, style, label, share=0.75):
    """
    Outline the region holding `share` of a set of points.

    Bins the points, smooths the counts, and draws the single contour that encloses
    that share of them. One line rather than several, because it sits on top of a
    colour scale that is already carrying information.

    Returns
    -------
    Line2D
        A proxy, so the contour can appear in a legend.
    """
    from scipy.ndimage import gaussian_filter

    xmin, xmax, ymin, ymax = extent
    counts, _, _ = np.histogram2d(x, y, bins=180, range=[[xmin, xmax], [ymin, ymax]])
    smoothed = gaussian_filter(counts, sigma=2.5)

    ordered = np.sort(smoothed.ravel())[::-1]
    cumulative = np.cumsum(ordered) / max(ordered.sum(), 1e-9)
    level = ordered[np.searchsorted(cumulative, share)]

    axis.contour(smoothed.T, levels=[level], colors=OUTLINE_COLOUR, linewidths=0.9,
                 linestyles=style, extent=extent)
    return mpl.lines.Line2D([], [], color=OUTLINE_COLOUR, lw=0.9, ls=style, label=label)


def property_maps(universe, coordinates):
    """
    The map coloured by each transport property, one figure per property.

    Each hexagon is the mean value of the molecules inside it, so no molecule hides
    another. Colours are clipped at the 2nd and 98th percentile: a handful of very large
    natural products would otherwise stretch the scale until every odorant is one shade.
    Yellow marks bins where no molecule has an estimate.
    """
    extent = (coordinates[:, 0].min(), coordinates[:, 0].max(),
              coordinates[:, 1].min(), coordinates[:, 1].max())
    in_lists = ((universe.in_leffingwell + universe.in_goodscents) > 0).values
    in_m2or = (universe.in_m2or == 1).values

    for column, label, name in PROPERTIES:
        figure, axis = plt.subplots(figsize=FIGSIZE)
        values = pd.to_numeric(universe[column], errors="coerce").values.astype(float)
        low, high = np.nanpercentile(values, [2, 98])
        known = np.isfinite(values)

        if (~known).any():
            axis.hexbin(coordinates[~known, 0], coordinates[~known, 1], gridsize=110,
                        mincnt=1, extent=extent, linewidths=0, rasterized=True,
                        cmap=mpl.colors.ListedColormap([NO_ESTIMATE_COLOUR]))
        bins = axis.hexbin(coordinates[known, 0], coordinates[known, 1], C=values[known],
                           reduce_C_function=np.mean, gridsize=110, mincnt=3,
                           extent=extent, cmap=PROPERTY_RAMP, vmin=low, vmax=high,
                           linewidths=0, rasterized=True)
        bar = figure.colorbar(bins, ax=axis, fraction=0.046, pad=0.02, extend="both")
        bar.set_label(label)
        bar.outline.set_visible(False)

        handles = [
            density_contour(axis, coordinates[in_lists, 0], coordinates[in_lists, 1],
                            extent, "-", f"Leffingwell / GoodScents "
                                         f"(n={int(in_lists.sum()):,})"),
            density_contour(axis, coordinates[in_m2or, 0], coordinates[in_m2or, 1],
                            extent, "--", f"M2OR (n={int(in_m2or.sum()):,})"),
        ]
        if (~known).any():
            handles.append(mpl.patches.Patch(facecolor=NO_ESTIMATE_COLOUR,
                                             label=f"no estimate "
                                                   f"({int((~known).sum()):,})"))
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")
        axis.legend(handles=handles, frameon=False, loc="upper center",
                    bbox_to_anchor=(0.5, -0.12), fontsize=8, handlelength=2.0)
        save(figure, name)


def organism_kingdom_by_list():
    """
    organism_kingdom repeated for each odorant list: all, Leffingwell/GoodScents, M2OR.

    Each list is compared against the same non-odorant natural products, with the
    permutation test and bootstrap interval of c07. The numbers on each bar pair are the
    ratio of the two mean shares; "n.s." marks q >= 0.05.
    """
    ranks = pd.read_csv(RESULTS / "organism_rank_enrichment.csv")
    everything = pd.concat([
        ranks.query("rank == 'kingdom' and taxon in @KINGDOM_ORDER"),
        ranks.query("rank == 'superkingdom' and taxon == 'Bacteria'"),
    ]).set_index("taxon")
    panels = [("all odorants", everything.mean_share_odorant,
               everything.mean_share_background, everything.share_ratio, everything.q,
               int(everything.n_odorant_molecules.iloc[0]),
               int(everything.n_background_molecules.iloc[0]))]

    to_ncbi = {"plants": "Viridiplantae", "animals": "Metazoa", "fungi": "Fungi",
               "bacteria": "Bacteria"}
    for label, stem in [("Leffingwell / GoodScents", "leffingwell_goodscents"),
                        ("M2OR", "m2or")]:
        path = RESULTS / f"{stem}_kingdom_stats.csv"
        if not path.exists():
            continue
        stats = pd.read_csv(path)
        stats.index = stats.kingdom.map(to_ncbi)
        panels.append((label, stats.mean_share_panel, stats.mean_share_background,
                       stats.fold_change, stats.q, int(stats.n_panel.iloc[0]),
                       int(stats.n_background.iloc[0])))

    figure, axes = plt.subplots(1, len(panels),
                                figsize=(len(panels) * FIGSIZE[0], FIGSIZE[1]),
                                sharey=True)
    for axis, (label, odorant, other, ratio, q, n_list, n_other) in zip(axes, panels):
        odorant_pct = np.array([100 * odorant[k] for k in KINGDOM_ORDER])
        other_pct = np.array([100 * other[k] for k in KINGDOM_ORDER])
        positions = paired_bars(axis, [KINGDOM_LABEL[k] for k in KINGDOM_ORDER],
                                odorant_pct, other_pct, f"{label} (n={n_list:,})",
                                f"other natural products (n={n_other:,})")
        for position, a, b, taxon in zip(positions, odorant_pct, other_pct, KINGDOM_ORDER):
            note = "" if q[taxon] < 0.05 else " n.s."
            axis.text(max(a, b) + 2, position, f"{ratio[taxon]:.2f}×{note}",
                      va="center", **ANNOTATION)
        axis.set_xlim(0, 115)
        axis.set_xticks([0, 25, 50, 75, 100])
        axis.set_xlabel("share of source organisms (%)")
        axis.set_title(label)
        axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22),
                    fontsize=8)
    figure.tight_layout()
    save(figure, "organism_kingdom_by_list")


def receptor_clusters_on_map(join, coordinates):
    """
    The six co-tuning clusters on the map, one facet each, beside c11's footprint ratios.

    A ratio near 1 means a cluster's members are as far from each other on the map as
    from the other clusters -- interleaved. Descriptive only; map distances are not metric.
    """
    footprint = pd.read_csv(RESULTS / "receptor_cluster_footprint.csv")
    sample = np.random.default_rng(0).choice(len(coordinates),
                                             min(120_000, len(coordinates)), replace=False)
    x_range = np.percentile(coordinates[:, 0], [0.2, 99.8])
    y_range = np.percentile(coordinates[:, 1], [0.2, 99.8])

    figure = plt.figure(figsize=(4 * FIGSIZE[0], 2 * FIGSIZE[1]))
    grid = figure.add_gridspec(2, 4, width_ratios=[1, 1, 1, 1.25], hspace=0.25,
                               wspace=0.15)
    for cluster in range(6):
        axis = figure.add_subplot(grid[cluster // 3, cluster % 3])
        axis.scatter(coordinates[sample, 0], coordinates[sample, 1], s=0.3,
                     c=FAINT_COLOUR, alpha=0.5, lw=0, rasterized=True)
        members = join.cluster == cluster
        axis.scatter(join.x[members], join.y[members], s=5, c=ODORANT_COLOUR,
                     alpha=0.85, lw=0, rasterized=True)
        axis.set_xlim(*x_range)
        axis.set_ylim(*y_range)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"cluster {cluster}  (n={int(members.sum())})",
                       color=CLUSTER_COLOURS[cluster])

    axis = figure.add_subplot(grid[:, 3])
    positions = np.arange(len(footprint))
    axis.barh(positions + 0.19, footprint["between"], height=0.36,
              color=BACKGROUND_COLOUR, label="to other clusters")
    axis.barh(positions - 0.19, footprint["within"], height=0.36,
              color=ODORANT_COLOUR, label="within cluster")
    for position, row in zip(positions, footprint.itertuples()):
        axis.text(max(row.within, row.between) + 0.08, position, f"{row.ratio:.2f}",
                  va="center", **ANNOTATION)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"cluster {c}" for c in footprint.cluster])
    axis.invert_yaxis()
    axis.set_xlabel("median pairwise map distance")
    axis.set_title("within / between (1 = interleaved)")
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2)
    axis.tick_params(left=False)
    save(figure, "receptor_clusters_on_map")


def receptor_clusters_properties(join):
    """Volatility and natural-product likeness per cluster, with c11's Kruskal-Wallis."""
    figure, axes = plt.subplots(1, 2, figsize=(2 * FIGSIZE[0], FIGSIZE[1]))
    for axis, (column, label) in zip(axes, [("Tb_joback", "boiling point (K)"),
                                            ("np_likeness", "natural-product likeness")]):
        groups = [join.loc[join.cluster == c, column].dropna().values for c in range(6)]
        statistic, p_value = kruskal(*groups)
        for cluster, values in enumerate(groups):
            jitter = cluster + np.random.default_rng(cluster).uniform(-0.16, 0.16,
                                                                      len(values))
            axis.scatter(jitter, values, s=5, color=CLUSTER_COLOURS[cluster], alpha=0.55,
                         lw=0, rasterized=True)
            low, middle, high = np.percentile(values, [25, 50, 75])
            axis.plot([cluster - 0.3, cluster + 0.3], [middle] * 2, color=GREY, lw=1.6,
                      solid_capstyle="round", zorder=5)
            axis.plot([cluster, cluster], [low, high], color=GREY, lw=0.8, zorder=4)
        axis.set_xticks(range(6))
        axis.set_xticklabels([f"{c}\nn={len(g)}" for c, g in enumerate(groups)],
                             fontsize=8)
        axis.set_xlabel("co-tuning cluster")
        axis.set_ylabel(label)
        axis.set_title(f"Kruskal-Wallis H = {statistic:.1f}, $P$ = {p_value:.2g}",
                       fontsize=9)
        axis.tick_params(axis="x", length=0)
    figure.tight_layout()
    save(figure, "receptor_clusters_properties")


def working_figures():
    """Every working figure."""
    universe = load_universe()
    coordinates = load_umap()
    property_maps(universe, coordinates)
    organism_kingdom_by_list()
    join = receptor_panel(universe, coordinates)
    receptor_clusters_on_map(join, coordinates)
    receptor_clusters_properties(join)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["paper", "working"], default=None,
                        help="draw only the paper figures, or only the working ones")
    args = parser.parse_args()

    if args.only in (None, "paper"):
        figure4()
        supplementary()
    if args.only in (None, "working"):
        working_figures()


if __name__ == "__main__":
    main()
