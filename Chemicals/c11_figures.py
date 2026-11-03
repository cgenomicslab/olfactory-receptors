"""Draw every figure, one SVG per panel, in the shared style of scripts/plotting_functions.py.

    python c11_figures.py                  # everything
    python c11_figures.py --only paper     # or --only working

Each panel is drawn by its own function, which takes data already loaded from results/ and
returns the matplotlib figure. A notebook can draw one panel at a time:

    import c11_figures as f
    figure = f.functional_groups_matched(f.load_table("fg_enrichment.csv"))

Figure 4
    a  odorant_map(universe, coordinates)              odorant_map.svg
    b  organism_kingdom(ranks)                         organism_kingdom.svg
    c  biosynthetic_pathway(pathways)                  biosynthetic_pathway.svg
    d  functional_groups_matched(groups)               functional_groups_matched.svg

Extended Data Fig. 7
    a  aromatic_map(universe, coordinates)             aromatic_map.svg
    b  aromatic_vs_aliphatic(universe)                 aromatic_vs_aliphatic.svg
    c  receptor_class_map(join, coordinates, ...)      receptor_class_map_class_I.svg
    d                                                  receptor_class_map_shared.svg
    e                                                  receptor_class_map_class_II.svg
    f  receptor_class_volatility(universe, join)       receptor_class_volatility.svg
    g  organism_rank(ranks, "order")                   organism_order.svg
    h  organism_rank(ranks, "family")                  organism_family.svg

Working figures: map_boiling_point, map_molecular_weight, map_logP, organism_kingdom_<list>
(all, leffingwell_goodscents, m2or), receptor_cluster_<k>_on_map (k = 0-5),
receptor_cluster_footprint, receptor_clusters_Tb_joback, receptor_clusters_np_likeness.

Inputs, all in results/: universe.parquet, embed_index.csv, umap.npy,
organism_rank_enrichment.csv, pathway_enrichment.csv, fg_enrichment.csv,
panel_universe_join.csv, receptor_cluster_footprint.csv, <list>_kingdom_stats.csv.
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
# 95% intervals, the same on every bar plot.
ERROR_BARS = {"fmt": "none", "ecolor": GREY, "elinewidth": 0.8, "capsize": 2}

# The four source groups and the NCBI rank each is taken at. Bacteria are the domain, not
# one of NCBI's two kingdom-rank halves of it.
KINGDOM_ORDER = ["Viridiplantae", "Metazoa", "Fungi", "Bacteria"]
KINGDOM_RANK = {"Viridiplantae": "kingdom", "Metazoa": "kingdom", "Fungi": "kingdom",
                "Bacteria": "domain"}
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
    "Fabales": "legumes", "Ranunculales": "poppy, buttercup", "Malpighiales": "willow, spurge",
    "Kitasatosporales": "Streptomyces", "Eurotiales": "Aspergillus, Penicillium",
    "Apiaceae": "carrot, cumin", "Brassicaceae": "mustard", "Solanaceae": "tomato",
    "Fabaceae": "legumes", "Apocynaceae": "periwinkle", "Streptomycetaceae": "Streptomyces",
    "Euphorbiaceae": "spurge",
}

# Host taxa (milk, breath, body odour) and the minimum share are defined once, in c07.
from c07_organism_source import HOST_TAXA, MIN_SHARE  # noqa: E402

CLASS_GROUPS = [("Class I only", "class_I", CLASS_I_COLOUR),
                ("shared", "shared", SHARED_COLOUR),
                ("Class II only", "class_II", CLASS_II_COLOUR)]


# ---------------------------------------------------------------------- loading
def load_table(name):
    """Read one CSV from results/, e.g. load_table("fg_enrichment.csv")."""
    return pd.read_csv(RESULTS / name)


def load_universe():
    """The universe, ordered to match the embedding and the map coordinates."""
    universe = pd.read_parquet(RESULTS / "universe.parquet")
    index = pd.read_csv(RESULTS / "embed_index.csv")
    return cs.align_to_embedding(universe, index)


def load_umap():
    """Map coordinates from c04, rows in the order of load_universe()."""
    return np.load(RESULTS / "umap.npy")


def receptor_panel(universe, coordinates, join):
    """
    The panel molecules c09 placed on the map, with map position and two properties.

    Parameters
    ----------
    universe : DataFrame
        From load_universe().
    coordinates : ndarray, shape (n_molecules, 2)
        From load_umap().
    join : DataFrame
        results/panel_universe_join.csv.

    Returns
    -------
    DataFrame
        The joined molecules, with x, y, Tb_joback and np_likeness added.
    """
    join = join[join.in_universe].copy()
    rows = join.row.astype(int).values
    join["x"] = coordinates[rows, 0]
    join["y"] = coordinates[rows, 1]
    join["Tb_joback"] = pd.to_numeric(universe.Tb_joback.values[rows], errors="coerce")
    join["np_likeness"] = pd.to_numeric(universe.np_likeness.values[rows], errors="coerce")
    return join


def kingdom_shares(ranks):
    """
    The four source groups' rows of c07's table, in KINGDOM_ORDER.

    Parameters
    ----------
    ranks : DataFrame
        results/organism_rank_enrichment.csv.

    Returns
    -------
    DataFrame
        Indexed by NCBI taxon name.
    """
    rows = [ranks[(ranks["rank"] == KINGDOM_RANK[taxon]) & (ranks.taxon == taxon)]
            for taxon in KINGDOM_ORDER]
    shares = pd.concat(rows).set_index("taxon")
    missing = [taxon for taxon in KINGDOM_ORDER if taxon not in shares.index]
    if missing:
        raise ValueError(f"organism_rank_enrichment.csv has no row for {missing}")
    return shares.loc[KINGDOM_ORDER]


# ---------------------------------------------------------------------- helpers
def save(figure, name):
    """Write one figure to figures/<name>.svg and close it."""
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.svg")
    plt.close(figure)
    print(f"wrote figures/{name}.svg", flush=True)


def blank_map_axes(axis):
    """UMAP axes carry no units, so no ticks; equal aspect so neighbourhoods keep shape.

    The axes box shrinks to the square the map needs, so the map fills it and the space
    beside it is left for the legend.
    """
    axis.set_aspect("equal", adjustable="box")
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


def map_background(axis, coordinates):
    """A fixed random 120,000 of the universe in faint grey, for maps with few points."""
    sample = np.random.default_rng(0).choice(len(coordinates),
                                             min(120_000, len(coordinates)), replace=False)
    axis.scatter(coordinates[sample, 0], coordinates[sample, 1], s=0.3, c=FAINT_COLOUR,
                 alpha=0.4, lw=0, rasterized=True)


# ------------------------------------------------------------------- Figure 4
def odorant_map(universe, coordinates):
    """
    Fig. 4a. Odorants on the natural-product map, with the M2OR subset picked out.

    M2OR points are drawn slightly larger than the other odorants so they stay findable,
    but small enough not to hide the wider odorant set beneath them.

    Parameters
    ----------
    universe : DataFrame
        From load_universe().
    coordinates : ndarray, shape (n_molecules, 2)
        From load_umap().

    Returns
    -------
    matplotlib.figure.Figure
    """
    is_odorant = universe.is_odorant.values == 1
    in_m2or = (universe.in_m2or == 1).values
    background = np.flatnonzero(~is_odorant)
    others = np.flatnonzero(is_odorant & ~in_m2or)
    m2or = np.flatnonzero(in_m2or)

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    axis.scatter(coordinates[background, 0], coordinates[background, 1], s=0.5,
                 c=BACKGROUND_COLOUR, alpha=0.3, lw=0, rasterized=True,
                 label=f"natural products\n({len(background):,})")
    axis.scatter(coordinates[others, 0], coordinates[others, 1], s=0.3,
                 c=ODORANT_COLOUR, alpha=0.55, lw=0, rasterized=True,
                 label=f"odorants\n({len(others):,})")
    axis.scatter(coordinates[m2or, 0], coordinates[m2or, 1], s=0.6, c=M2OR_COLOUR,
                 alpha=0.8, lw=0, rasterized=True, zorder=4,
                 label=f"M2OR\n({len(m2or):,})")
    blank_map_axes(axis)
    figure.legend(frameon=False, markerscale=6, loc="outside right upper", fontsize=8,
                  labelspacing=1.0, handletextpad=0.4)
    return figure


def organism_kingdom(ranks):
    """
    Fig. 4b. Mean share of a molecule's source organisms in each group, from c07.

    Parameters
    ----------
    ranks : DataFrame
        results/organism_rank_enrichment.csv.

    Returns
    -------
    matplotlib.figure.Figure
    """
    shares = kingdom_shares(ranks)

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
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
    return figure


def biosynthetic_pathway(pathways):
    """
    Fig. 4c. NPClassifier pathway, as a percentage of the annotated molecules on each side.

    Parameters
    ----------
    pathways : DataFrame
        results/pathway_enrichment.csv.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = pathways.sort_values("pct_odorant", ascending=False)
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    paired_bars(axis, [p.replace(" and ", "\n") for p in table.pathway],
                table.pct_odorant.values, table.pct_background.values,
                "odorants", "natural products")
    axis.set_xlabel("% of annotated molecules")
    axis.legend(frameon=False, loc="lower right")
    return figure


def functional_groups_matched(groups):
    """
    Fig. 4d. Functional groups as log2 odds ratio, odorants against their matched partners.

    Whiskers are the 95% interval from resampling pairs (c05); groups with McNemar
    BH q >= 0.05 are drawn pale.

    Parameters
    ----------
    groups : DataFrame
        results/fg_enrichment.csv.

    Returns
    -------
    matplotlib.figure.Figure
    """
    table = groups.sort_values("OR_matched")
    positions = np.arange(len(table))
    log_odds = np.log2(table.OR_matched.values)
    low = np.log2(table.ci_lo_matched.values)
    high = np.log2(table.ci_hi_matched.values)
    significant = table.q_matched.values < 0.05

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    colours = [mpl.colors.to_rgba(ODORANT_BAR_COLOUR if value > 0 else BACKGROUND_COLOUR,
                                  1.0 if is_significant else 0.35)
               for value, is_significant in zip(log_odds, significant)]
    axis.barh(positions, log_odds, color=colours, height=0.68)
    axis.errorbar(log_odds, positions, xerr=[log_odds - low, high - log_odds], **ERROR_BARS)
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
    return figure


def figure4():
    """Draw and save the four panels of Figure 4."""
    universe = load_universe()
    save(odorant_map(universe, load_umap()), "odorant_map")
    save(organism_kingdom(load_table("organism_rank_enrichment.csv")), "organism_kingdom")
    save(biosynthetic_pathway(load_table("pathway_enrichment.csv")), "biosynthetic_pathway")
    save(functional_groups_matched(load_table("fg_enrichment.csv")),
         "functional_groups_matched")


# ----------------------------------------------------------- Extended Data Fig. 7
def aromatic_split(universe):
    """Masks of the odorants without and with an aromatic ring (RDKit AromaticRings)."""
    is_odorant = universe.is_odorant.values == 1
    aromatic_ring = universe["AromaticRings"].values > 0
    return is_odorant & ~aromatic_ring, is_odorant & aromatic_ring


def aromatic_map(universe, coordinates):
    """
    Extended Data Fig. 7a. The aromatic and aliphatic odorants on the map.

    Parameters
    ----------
    universe : DataFrame
        From load_universe().
    coordinates : ndarray, shape (n_molecules, 2)
        From load_umap().

    Returns
    -------
    matplotlib.figure.Figure
    """
    aliphatic, aromatic = aromatic_split(universe)
    percent = 100 * aliphatic.sum() / (aliphatic.sum() + aromatic.sum())
    background = np.flatnonzero(universe.is_odorant.values != 1)

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
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
    return figure


def aromatic_vs_aliphatic(universe):
    """
    Extended Data Fig. 7b. Standardised mean difference, aromatic against aliphatic odorants.

    FracCsp3 is left out: aromatic carbons are sp2, so its difference follows from the
    definition of the split.

    Parameters
    ----------
    universe : DataFrame
        From load_universe().

    Returns
    -------
    matplotlib.figure.Figure
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

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
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
    return figure


def receptor_class_map(join, coordinates, group, colour):
    """
    Extended Data Fig. 7c-e. One receptor-class group on the map.

    Parameters
    ----------
    join : DataFrame
        From receptor_panel().
    coordinates : ndarray, shape (n_molecules, 2)
        From load_umap(), for the background.
    group : str
        "Class I only", "shared" or "Class II only".
    colour : str
        Point colour.

    Returns
    -------
    matplotlib.figure.Figure
    """
    subset = join[join.class_group == group]
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    map_background(axis, coordinates)
    axis.scatter(subset.x, subset.y, s=4, c=colour, alpha=0.85, lw=0, rasterized=True)
    blank_map_axes(axis)
    axis.set_title(f"{group}  (n={len(subset)})", color=colour)
    return figure


def receptor_class_volatility(universe, join):
    """
    Extended Data Fig. 7f. Boiling point by receptor class, and of every other natural product.

    The title gives the Kruskal-Wallis P across the three classes and Cliff's delta of
    class I against the other natural products.

    Parameters
    ----------
    universe : DataFrame
        From load_universe().
    join : DataFrame
        From receptor_panel().

    Returns
    -------
    matplotlib.figure.Figure
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

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    for position, (values, colour) in enumerate(zip(groups, colours)):
        # A fixed sample of at most 800 points per strip keeps the background readable.
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
    return figure


def organism_rank(ranks, rank, n_shown=10):
    """
    Extended Data Fig. 7g (order) and 7h (family). Enriched and depleted source taxa.

    Taxa with q < 0.05 that account for at least MIN_SHARE of sources on both sides, host
    taxa left out; the n_shown / 2 most enriched and most depleted are drawn, with their
    bootstrap 95% intervals. Requiring the share on both sides keeps out taxa that are
    almost absent from one group, whose fold change is large only because it divides
    by nearly zero.

    Parameters
    ----------
    ranks : DataFrame
        results/organism_rank_enrichment.csv.
    rank : str
        An NCBI rank tested by c07, e.g. "order" or "family".
    n_shown : int, optional

    Returns
    -------
    matplotlib.figure.Figure
    """
    at_rank = ranks[(ranks["rank"] == rank) & (ranks.q < 0.05)].dropna(
        subset=["share_ratio"])
    at_rank = at_rank[~at_rank.taxon.isin(HOST_TAXA)]
    substantial = at_rank[
        at_rank[["mean_share_odorant", "mean_share_background"]].min(axis=1) >= MIN_SHARE
    ].sort_values("share_ratio", ascending=False)

    half = n_shown // 2
    shown = pd.concat([substantial.head(half), substantial.tail(half)])
    positions = np.arange(len(shown))[::-1]
    log_ratio = np.log2(shown.share_ratio.values)
    below = log_ratio - np.log2(shown.ratio_ci_lo.values)
    above = np.log2(shown.ratio_ci_hi.values) - log_ratio

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    axis.barh(positions, log_ratio, height=0.66,
              color=[ODORANT_COLOUR if v > 0 else BACKGROUND_COLOUR for v in log_ratio])
    axis.errorbar(log_ratio, positions, xerr=[below, above], **ERROR_BARS)
    axis.axvline(0, color=ZERO_LINE, lw=0.8)
    for position, value, ratio, low, high in zip(positions, log_ratio, shown.share_ratio,
                                                 below, above):
        # Two significant digits below 1, so a 25-fold depletion does not print as 0.0x.
        label = f"{ratio:.1f}×" if ratio >= 1 else f"{ratio:.2g}×"
        # Past the end of the interval, so the label does not sit on the error bar.
        end = value + high + 0.15 if value >= 0 else value - low - 0.15
        axis.text(end, position, label,
                  va="center", ha="left" if value >= 0 else "right", **ANNOTATION)
    axis.set_yticks(positions)
    axis.set_yticklabels([f"{t}\n{TAXON_EXAMPLES[t]}" if t in TAXON_EXAMPLES else t
                          for t in shown.taxon], fontsize=8)
    limit = max(abs(log_ratio - below).max(), abs(log_ratio + above).max()) * 1.6
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("log$_2$ fold change in source share")
    axis.set_title(rank.capitalize())
    axis.tick_params(left=False)
    return figure


def extended_data_7():
    """Draw and save the eight panels of Extended Data Fig. 7."""
    universe = load_universe()
    coordinates = load_umap()
    join = receptor_panel(universe, coordinates, load_table("panel_universe_join.csv"))
    ranks = load_table("organism_rank_enrichment.csv")
    save(aromatic_map(universe, coordinates), "aromatic_map")
    save(aromatic_vs_aliphatic(universe), "aromatic_vs_aliphatic")
    for group, stem, colour in CLASS_GROUPS:
        save(receptor_class_map(join, coordinates, group, colour),
             f"receptor_class_map_{stem}")
    save(receptor_class_volatility(universe, join), "receptor_class_volatility")
    save(organism_rank(ranks, "order"), "organism_order")
    save(organism_rank(ranks, "family"), "organism_family")


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


def property_map(universe, coordinates, column, label):
    """
    The map coloured by one transport property.

    Each hexagon is the mean value of the molecules inside it, so no molecule hides
    another. Colours are clipped at the 2nd and 98th percentile so a handful of very large
    natural products do not stretch the scale. Yellow marks bins where no molecule has an
    estimate.

    Returns
    -------
    matplotlib.figure.Figure
    """
    extent = (coordinates[:, 0].min(), coordinates[:, 0].max(),
              coordinates[:, 1].min(), coordinates[:, 1].max())
    in_lists = ((universe.in_leffingwell + universe.in_goodscents) > 0).values
    in_m2or = (universe.in_m2or == 1).values

    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
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
    return figure


def kingdom_comparisons(ranks, list_stats):
    """
    The kingdom comparison of c07 for each odorant list: all, Leffingwell/GoodScents, M2OR.

    Each list is compared against the same non-odorant natural products, with the
    permutation test and bootstrap interval of c07.

    Parameters
    ----------
    ranks : DataFrame
        results/organism_rank_enrichment.csv, for all odorants.
    list_stats : dict of str to DataFrame
        List label -> results/<list>_kingdom_stats.csv from c08.

    Returns
    -------
    list of tuple
        (label, odorant shares, other shares, ratio, q, n in the list, n other), one per list.
    """
    everything = kingdom_shares(ranks)
    comparisons = [("all odorants", everything.mean_share_odorant,
                    everything.mean_share_background, everything.share_ratio, everything.q,
                    int(everything.n_odorant_molecules.iloc[0]),
                    int(everything.n_background_molecules.iloc[0]))]
    to_ncbi = {KINGDOM_LABEL[taxon]: taxon for taxon in KINGDOM_ORDER}
    for label, stats in list_stats.items():
        stats = stats.copy()
        stats.index = stats.kingdom.map(to_ncbi)
        comparisons.append((label, stats.mean_share_panel, stats.mean_share_background,
                            stats.fold_change, stats.q, int(stats.n_panel.iloc[0]),
                            int(stats.n_background.iloc[0])))
    return comparisons


def organism_kingdom_for_list(label, odorant, other, ratio, q, n_list, n_other):
    """
    Source-organism shares by kingdom for one odorant list, against other natural products.

    The number beside each bar pair is the ratio of the two mean shares; "n.s." marks
    q >= 0.05.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    odorant_pct = np.array([100 * odorant[k] for k in KINGDOM_ORDER])
    other_pct = np.array([100 * other[k] for k in KINGDOM_ORDER])
    positions = paired_bars(axis, [KINGDOM_LABEL[k] for k in KINGDOM_ORDER],
                            odorant_pct, other_pct, f"{label} (n={n_list:,})",
                            f"other natural products (n={n_other:,})")
    for position, a, b, taxon in zip(positions, odorant_pct, other_pct, KINGDOM_ORDER):
        note = "" if q[taxon] < 0.05 else " n.s."
        axis.text(max(a, b) + 2, position, f"{ratio[taxon]:.2f}×{note}", va="center",
                  **ANNOTATION)
    axis.set_xlim(0, 115)
    axis.set_xticks([0, 25, 50, 75, 100])
    axis.set_xlabel("share of source organisms (%)")
    axis.set_title(label)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    return figure


def cluster_on_map(join, coordinates, cluster):
    """
    One co-tuning cluster's odorants on the map, over a fixed sample of the universe.

    Parameters
    ----------
    join : DataFrame
        From receptor_panel().
    coordinates : ndarray, shape (n_molecules, 2)
    cluster : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    sample = np.random.default_rng(0).choice(len(coordinates),
                                             min(120_000, len(coordinates)), replace=False)
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    axis.scatter(coordinates[sample, 0], coordinates[sample, 1], s=0.3,
                 c=FAINT_COLOUR, alpha=0.5, lw=0, rasterized=True)
    members = join.cluster == cluster
    axis.scatter(join.x[members], join.y[members], s=5, c=ODORANT_COLOUR,
                 alpha=0.85, lw=0, rasterized=True)
    axis.set_xlim(*np.percentile(coordinates[:, 0], [0.2, 99.8]))
    axis.set_ylim(*np.percentile(coordinates[:, 1], [0.2, 99.8]))
    blank_map_axes(axis)
    axis.set_title(f"cluster {cluster}  (n={int(members.sum())})", color=CLUSTER_COLOURS[cluster])
    return figure


def cluster_footprint(footprint):
    """
    c09's footprint ratios: median map distance within each cluster and to the others.

    A ratio near 1 means a cluster's members are as far from each other on the map as
    from the other clusters. Map distances are not metric, so this is descriptive.

    Parameters
    ----------
    footprint : DataFrame
        results/receptor_cluster_footprint.csv.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
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
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    axis.tick_params(left=False)
    return figure


def cluster_property(join, column, label):
    """
    One property per co-tuning cluster, with the Kruskal-Wallis test across clusters.

    Parameters
    ----------
    join : DataFrame
        From receptor_panel().
    column : str
        "Tb_joback" or "np_likeness".
    label : str
        Axis label.

    Returns
    -------
    matplotlib.figure.Figure
    """
    figure, axis = plt.subplots(figsize=FIGSIZE, layout="constrained")
    groups = [join.loc[join.cluster == c, column].dropna().values for c in range(6)]
    statistic, p_value = kruskal(*groups)
    for cluster, values in enumerate(groups):
        jitter = cluster + np.random.default_rng(cluster).uniform(-0.16, 0.16, len(values))
        axis.scatter(jitter, values, s=5, color=CLUSTER_COLOURS[cluster], alpha=0.55,
                     lw=0, rasterized=True)
        low, middle, high = np.percentile(values, [25, 50, 75])
        axis.plot([cluster - 0.3, cluster + 0.3], [middle] * 2, color=GREY, lw=1.6,
                  solid_capstyle="round", zorder=5)
        axis.plot([cluster, cluster], [low, high], color=GREY, lw=0.8, zorder=4)
    axis.set_xticks(range(6))
    axis.set_xticklabels([f"{c}\nn={len(g)}" for c, g in enumerate(groups)], fontsize=8)
    axis.set_xlabel("co-tuning cluster")
    axis.set_ylabel(label)
    axis.set_title(f"Kruskal-Wallis H = {statistic:.1f}, $P$ = {p_value:.2g}", fontsize=9)
    axis.tick_params(axis="x", length=0)
    return figure


def working_figures():
    """Draw and save every working figure."""
    universe = load_universe()
    coordinates = load_umap()
    for column, label, name in PROPERTIES:
        save(property_map(universe, coordinates, column, label), name)

    list_stats = {}
    for label, stem in [("Leffingwell / GoodScents", "leffingwell_goodscents"),
                        ("M2OR", "m2or")]:
        path = RESULTS / f"{stem}_kingdom_stats.csv"
        if path.exists():
            list_stats[label] = pd.read_csv(path)
    file_stem = {"all odorants": "all", "Leffingwell / GoodScents": "leffingwell_goodscents", "M2OR": "m2or"}
    for comparison in kingdom_comparisons(load_table("organism_rank_enrichment.csv"), list_stats):
        save(organism_kingdom_for_list(*comparison), f"organism_kingdom_{file_stem[comparison[0]]}")

    join = receptor_panel(universe, coordinates, load_table("panel_universe_join.csv"))
    for cluster in range(6):
        save(cluster_on_map(join, coordinates, cluster), f"receptor_cluster_{cluster}_on_map")
    save(cluster_footprint(load_table("receptor_cluster_footprint.csv")), "receptor_cluster_footprint")
    for column, label in [("Tb_joback", "boiling point (K)"), ("np_likeness", "natural-product likeness")]:
        save(cluster_property(join, column, label), f"receptor_clusters_{column}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=["paper", "working"], default=None,
                        help="draw only the paper figures, or only the working ones")
    args = parser.parse_args()

    if args.only in (None, "paper"):
        figure4()
        extended_data_7()
    if args.only in (None, "working"):
        working_figures()


if __name__ == "__main__":
    main()
