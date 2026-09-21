"""Step 5 -- the figures. Reads only what s01-s04 and s06 wrote.

  F0  The molecules in PCA space, where distances are real, plus the variance
      each component carries
  F1  Odorants on the natural-product map, and how clustered they are
  F3  What an odorant is made of: biosynthetic pathway and functional groups
  F4  The split between odorants with and without a benzene ring
  F6  The map coloured by each transport feature: F6a boiling point,
      F6b molecular weight, F6c logP
  F7  Which kingdoms odorants are isolated from
  S1  The same, at order and family level (supplementary)

The comparison throughout is odorants against the whole natural-product background.
That is what COCONUT was chosen for -- the differences between the two are the result,
not a confound to be controlled away.

Colours are the project's palm-green and sand pair, checked for colour-vision separation
(deltaE 22.7 protan, 28.6 normal). No red: the human landscape figures already use it.

Any figure whose inputs are missing is skipped with a message, so this is safe to run at
any point in the pipeline.

CPU only, about two minutes. Writes SVG into figures/.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"          # overridden by --results
FIGURES = HERE / "figures"

ODORANT_COLOUR = "#0b7350"          # palm green
BACKGROUND_COLOUR = "#bfae94"       # sand
AROMATIC_COLOUR = "#8a6a3b"
# M2OR gets one colour across every figure it appears in, so the same colour always
# means the same set of molecules. Checked against the sand and palm-green it sits
# beside: deltaE 25.0 normal vision, 17.1 deutan, 16.3 tritan.
M2OR_COLOUR = "#5b2c83"
FAINT_COLOUR = "#e9e5db"
INK, MUTED, GRID = "#3f3f3c", "#8a8a86", "#e8e8e4"

# How many background points to draw on the maps. All 720k would be a solid block of
# colour and a very slow SVG, and the shape is already clear at this many.
MAP_SAMPLE = 120_000

mpl.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 170,
    # Keep text as text in the exported files, so labels stay selectable and editable
    # when the SVG is opened in Illustrator. Same convention as scripts/plotting_functions.py.
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "axes.edgecolor": "#bbbbb8", "axes.labelcolor": MUTED,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
})


def save(figure, name):
    """Write one figure as SVG."""
    FIGURES.mkdir(exist_ok=True)
    figure.savefig(FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(figure)
    print(f"wrote figures/{name}.svg", flush=True)


# Which map the figures draw, and what gets appended to their filenames. Set by --map.
# Keeping both means the cheaper PCA-50 version stays on disk for reference instead of
# being silently overwritten the moment the expensive one appears.
MAP_CHOICE = "raw"
MAP_SUFFIX = ""


def load_umap():
    """
    Read the map coordinates, either version.

    s03 writes umap.npy from the 50 principal components and umap_raw768.npy from the
    full embedding (--raw). Every figure that draws a map reads through here, so the
    whole set switches together and no two figures can show different layouts.
    """
    if MAP_CHOICE == "raw":
        raw = RESULTS / "umap_raw768.npy"
        if raw.exists():
            print("  map: umap_raw768.npy (all 768 dimensions)", flush=True)
            return np.load(raw)
        print("  map: umap_raw768.npy missing, falling back to PCA-50", flush=True)
    print("  map: umap.npy (50 principal components)", flush=True)
    return np.load(RESULTS / "umap.npy")


def load_universe():
    """Read the universe, ordered to match the embedding and the map coordinates."""
    universe = pd.read_parquet(RESULTS / "universe.parquet")
    index = pd.read_csv(RESULTS / "embed_index.csv")
    return universe.set_index("inchikey").loc[index.inchikey].reset_index()


def as_percentage(column):
    """Some files store fractions and some percentages; return percentages either way."""
    values = pd.to_numeric(column, errors="coerce")
    return values * 100 if values.max() <= 1.0 else values


def background_sample(n_points):
    """A reproducible random subset of row numbers, for the map backgrounds."""
    return np.random.default_rng(0).choice(n_points, min(MAP_SAMPLE, n_points),
                                           replace=False)


def load_pathways():
    """Read the pathway table, accepting either of the two column namings."""
    table = pd.read_csv(RESULTS / "pathway_enrichment.csv")
    pathways = pd.DataFrame({"pathway": table["pathway"]})
    pathways["pct_odorant"] = as_percentage(
        table["pct_odorant" if "pct_odorant" in table else "odorant_frac"]
    )
    pathways["pct_background"] = as_percentage(
        table["pct_background" if "pct_background" in table else "bg_frac"]
    )
    pathways["odds_ratio"] = table["odds_ratio"]
    pathways["q"] = table["q"] if "q" in table else table["p"]
    return pathways


def load_functional_groups(matched=False):
    """
    Read the functional-group table.

    The primary comparison is odorants against the whole natural-product background.
    That is the question COCONUT was chosen to answer -- how odour chemistry differs
    from natural-product chemistry -- and the differences are the result, not a
    confound to be removed.

    The matched columns answer a narrower follow-up: whether anything survives once
    volatility and size are equalised. Useful for one paragraph, but it partly erases
    the very thing that makes a molecule an odorant, so it is not the default.

    Parameters
    ----------
    matched : bool, optional
        Read the matched contrast instead of the raw one.
    """
    table = pd.read_csv(RESULTS / "fg_enrichment.csv")
    groups = pd.DataFrame({"group": table["group"]})

    if matched and "OR_matched" in table:
        groups["pct_odorant"] = as_percentage(table["odor_matched"])
        groups["pct_background"] = as_percentage(table["bg_matched"])
        groups["odds_ratio"] = table["OR_matched"]
        groups["q"] = table["q_matched"] if "q_matched" in table else table["p_matched"]
        groups.attrs["matched"] = True
    else:
        groups["pct_odorant"] = as_percentage(table["pct_odorant"])
        groups["pct_background"] = as_percentage(table["pct_background"])
        groups["odds_ratio"] = table["odds_ratio"]
        groups["q"] = table["q"] if "q" in table else table["p"]
        groups.attrs["matched"] = False
    return groups


def standardised_difference(first, second):
    """Difference between two means, in pooled standard deviations. Unitless."""
    first = first[np.isfinite(first)]
    second = second[np.isfinite(second)]
    pooled_sd = np.sqrt((np.nanvar(first) + np.nanvar(second)) / 2)
    if pooled_sd <= 0:
        return 0.0
    return (np.nanmean(first) - np.nanmean(second)) / pooled_sd


# ----------------------------------------------------------------------- F0 PCA
def figure0_pca(universe):
    """
    The molecules in principal-component space, and how much variance each axis holds.

    Worth having next to the UMAP because the two are trustworthy in different ways.
    PCA is a rotation: the distance between two points on this plot is a real distance,
    and each axis is a fixed direction that means the same thing everywhere. UMAP
    preserves roughly who is near whom but its distances and axes carry no such meaning.
    So anything read off positions should be read here, and the UMAP kept for showing
    neighbourhood structure.

    The cost is that PC1 and PC2 together hold only about a quarter of the variance, so
    two molecules sitting on top of each other here may still differ in the other 766
    directions.
    """
    variance_file = RESULTS / "pca_evr.npy"
    coordinates_file = RESULTS / "pca50.npy"
    if not (variance_file.exists() and coordinates_file.exists()):
        print("no pca50.npy / pca_evr.npy; skipping F0", flush=True)
        return

    per_component = np.load(variance_file)
    coordinates = np.load(coordinates_file)
    is_odorant = universe.is_odorant.values == 1

    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.8),
                                gridspec_kw={"width_ratios": [1, 1.25]})

    # A -- how much of the variance each direction carries.
    axis = axes[0]
    component_numbers = np.arange(1, len(per_component) + 1)
    axis.bar(component_numbers, 100 * per_component, color=BACKGROUND_COLOUR, width=0.85)
    axis.set_xlabel("principal component")
    axis.set_ylabel("% variance explained")
    axis.set_title(f"A  Variance per component\n"
                   f"PC1 {100 * per_component[0]:.1f}%, "
                   f"first 50 together {100 * per_component.sum():.1f}%", color=INK)

    # B -- the molecules themselves, on the first two components.
    #
    # Both groups are drawn whole and at the same point size, so what the eye sees is
    # the real density of each. Subsampling the background, or enlarging the odorants,
    # would make the odorant set look denser than it is.
    # M2OR is drawn separately and slightly larger. It is the set with receptor data
    # behind it, so it is worth being able to see where those particular molecules sit
    # relative to the odour lists as a whole.
    axis = axes[1]
    in_m2or = (universe.in_m2or == 1).values
    background_rows = np.flatnonzero(~is_odorant)
    other_odorant_rows = np.flatnonzero(is_odorant & ~in_m2or)
    m2or_rows = np.flatnonzero(in_m2or)

    axis.scatter(coordinates[background_rows, 0], coordinates[background_rows, 1],
                 s=1.2, c=BACKGROUND_COLOUR, alpha=0.25, lw=0, rasterized=True,
                 label=f"natural products (n={len(background_rows):,})")
    axis.scatter(coordinates[other_odorant_rows, 0], coordinates[other_odorant_rows, 1],
                 s=1.2, c=ODORANT_COLOUR, alpha=0.55, lw=0, rasterized=True,
                 label=f"odorants, not in M2OR (n={len(other_odorant_rows):,})")
    axis.scatter(coordinates[m2or_rows, 0], coordinates[m2or_rows, 1],
                 s=1.5, c=M2OR_COLOUR, alpha=0.8, lw=0, rasterized=True, zorder=4,
                 label=f"M2OR odorants (n={len(m2or_rows):,})")

    # The zero lines. Unlike on a UMAP, these mean something: PCA is centred, so the
    # origin is the average molecule and each axis is a fixed direction away from it.
    # Drawn on top of the points: underneath, the 726k rasterized dots paint over them.
    axis.axhline(0, color=GRID, lw=1, ls="--", zorder=5)
    axis.axvline(0, color=GRID, lw=1, ls="--", zorder=5)

    # Equal aspect, because the caption claims these distances are real. On unequal
    # axes a step along PC1 would look longer or shorter than the same step along PC2.
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel(f"PC1  ({100 * per_component[0]:.1f}% of variance)")
    axis.set_ylabel(f"PC2  ({100 * per_component[1]:.1f}%)")
    axis.set_title("B  The molecules on PC1 and PC2\n"
                   "(equal axes; distances here are real, unlike on the UMAP)",
                   color=INK)
    axis.legend(frameon=False, fontsize=8, markerscale=6, loc="upper left",
                labelcolor=INK)

    figure.tight_layout()
    save(figure, "F0_pca")


# ------------------------------------------------------------------------ F1 map
def figure1_map(universe):
    """
    Odorants drawn on the natural-product map.

    Illustrative only. How clustered odorants actually are is measured in the full
    768-dimensional space by s04 and s06, never from these coordinates -- UMAP
    preserves roughly who is near whom, not how far apart anything is.
    """
    coordinates = load_umap()
    is_odorant = universe.is_odorant.values == 1

    figure, axis = plt.subplots(figsize=(7.0, 5.8))

    # Same three layers, colours and point sizes as F0, so the two maps can be read
    # against each other without the eye having to re-learn what anything means. Every
    # group is drawn whole, so the relative density on the page is the real one.
    in_m2or = (universe.in_m2or == 1).values
    background_rows = np.flatnonzero(~is_odorant)
    other_odorant_rows = np.flatnonzero(is_odorant & ~in_m2or)
    m2or_rows = np.flatnonzero(in_m2or)

    axis.scatter(coordinates[background_rows, 0], coordinates[background_rows, 1],
                 s=1.2, c=BACKGROUND_COLOUR, alpha=0.25, lw=0, rasterized=True,
                 label=f"natural products (n={len(background_rows):,})")
    axis.scatter(coordinates[other_odorant_rows, 0], coordinates[other_odorant_rows, 1],
                 s=1.2, c=ODORANT_COLOUR, alpha=0.55, lw=0, rasterized=True,
                 label=f"odorants, not in M2OR (n={len(other_odorant_rows):,})")
    axis.scatter(coordinates[m2or_rows, 0], coordinates[m2or_rows, 1],
                 s=1.5, c=M2OR_COLOUR, alpha=0.8, lw=0, rasterized=True, zorder=4,
                 label=f"M2OR odorants (n={len(m2or_rows):,})")

    # No tick values and no zero lines: UMAP coordinates are arbitrary. The layout is
    # fixed only up to a shift, a rotation and a flip, so zero lands wherever the
    # optimisation left it and moves with the seed. Numbers here would invite reading
    # that is not supported. F0 is the plot whose axes mean something.
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Odorants on the natural-product map\n"
                   "(MolFormer embedding; layout illustrative)", color=INK)
    axis.legend(frameon=False, fontsize=8.5, markerscale=4, loc="best", labelcolor=INK)

    figure.tight_layout()
    save(figure, "F1_np_map" + MAP_SUFFIX)


# ------------------------------------------------------------- F3 what it is made of
def figure3_composition(universe):
    """Biosynthetic pathway on the left, functional groups on the right."""
    figure, axes = plt.subplots(1, 2, figsize=(11.6, 4.6))

    axis = axes[0]
    pathways = load_pathways().sort_values("pct_odorant")
    positions = np.arange(len(pathways))
    axis.barh(positions + 0.2, pathways["pct_odorant"], height=0.38,
              color=ODORANT_COLOUR, label="odorants")
    axis.barh(positions - 0.2, pathways["pct_background"], height=0.38,
              color=BACKGROUND_COLOUR, label="natural products")
    for position, (_, row) in enumerate(pathways.iterrows()):
        stars = "***" if row.q < 1e-3 else "**" if row.q < 1e-2 else "*" if row.q < 0.05 else ""
        if stars:
            axis.text(max(row.pct_odorant, row.pct_background) + 1.2, position, stars,
                      va="center", fontsize=8, color=MUTED)
    axis.set_yticks(positions)
    axis.set_yticklabels([name.replace(" and ", " /\n") for name in pathways["pathway"]],
                         fontsize=7.5)
    axis.set_xlabel("% of annotated molecules")
    axis.set_title("A  Biosynthetic pathway\nwhat odorants are made of", color=INK)
    axis.legend(frameon=False, fontsize=8, loc="lower right", labelcolor=INK)
    axis.tick_params(left=False)

    axis = axes[1]
    groups = load_functional_groups().sort_values("odds_ratio")
    positions = np.arange(len(groups))
    colours = [ODORANT_COLOUR if value > 1 else BACKGROUND_COLOUR
               for value in groups["odds_ratio"]]
    axis.barh(positions, np.log2(groups["odds_ratio"].clip(lower=1e-3)),
              color=colours, height=0.66)
    axis.axvline(0, color=GRID, lw=1)
    axis.set_yticks(positions)
    axis.set_yticklabels(
        [name.replace("fg_", "").replace("_", " ") for name in groups["group"]], fontsize=8
    )
    axis.set_xlabel("log$_2$ odds ratio  (odorant vs natural product)")
    subtitle = ("after matching" if groups.attrs.get("matched")
                else "vs all natural products")
    axis.set_title(f"B  Functional groups\n{subtitle}", color=INK)
    axis.tick_params(left=False)

    figure.tight_layout()
    save(figure, "F3_what_an_odorant_is")


# ---------------------------------------------------------------- F4 the two halves
def figure4_aromatic_split(universe):
    """Odorants divide into those built around a benzene ring and those not."""
    coordinates = load_umap()
    is_odorant = universe.is_odorant.values == 1
    has_aromatic_ring = universe["AromaticRings"].values > 0

    aliphatic = is_odorant & ~has_aromatic_ring
    aromatic = is_odorant & has_aromatic_ring
    percent_aliphatic = 100 * aliphatic.sum() / is_odorant.sum()

    figure, axes = plt.subplots(1, 2, figsize=(11.2, 4.6))

    axis = axes[0]
    sample = background_sample(int((~is_odorant).sum()))
    background_rows = np.flatnonzero(~is_odorant)[sample]
    axis.scatter(coordinates[background_rows, 0], coordinates[background_rows, 1],
                 s=1.2, c=FAINT_COLOUR, alpha=0.35, lw=0, rasterized=True)
    axis.scatter(coordinates[aliphatic, 0], coordinates[aliphatic, 1], s=4,
                 c=ODORANT_COLOUR, alpha=0.8, lw=0, rasterized=True,
                 label=f"aliphatic ({percent_aliphatic:.0f}%)")
    axis.scatter(coordinates[aromatic, 0], coordinates[aromatic, 1], s=4,
                 c=AROMATIC_COLOUR, alpha=0.8, lw=0, rasterized=True,
                 label=f"aromatic ({100 - percent_aliphatic:.0f}%)")
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_xlabel("UMAP 1")
    axis.set_ylabel("UMAP 2")
    axis.set_title("Odorants split by the benzene ring", color=INK)
    axis.legend(frameon=False, fontsize=8, markerscale=3, loc="best", labelcolor=INK)

    # Standardised differences rather than ratios of medians: unitless, comparable across
    # descriptors, and they do not collapse to zero when two medians happen to tie.
    axis = axes[1]
    wanted = ["MolWt", "Tb_joback", "LogP", "TPSA", "nO", "FracCsp3", "RotBonds", "HBA"]
    names = [name for name in wanted if name in universe]
    differences = []
    for name in names:
        values = pd.to_numeric(universe[name], errors="coerce").values.astype(float)
        differences.append(standardised_difference(values[aromatic], values[aliphatic]))

    order = np.argsort(differences)
    names = [names[i] for i in order]
    differences = [differences[i] for i in order]

    positions = np.arange(len(names))
    axis.barh(positions, differences,
              color=[AROMATIC_COLOUR if v > 0 else ODORANT_COLOUR for v in differences],
              height=0.62)
    axis.axvline(0, color=GRID, lw=1)
    for position, value in enumerate(differences):
        axis.text(value + (0.03 if value >= 0 else -0.03), position, f"{value:+.2f}",
                  va="center", ha="left" if value >= 0 else "right",
                  fontsize=7.5, color=MUTED)
    axis.set_yticks(positions)
    axis.set_yticklabels(names, fontsize=8.5)
    limit = max(abs(min(differences)), abs(max(differences))) * 1.35
    axis.set_xlim(-limit, limit)
    axis.set_xlabel("standardised mean difference\n"
                    "(<- aliphatic higher   |   aromatic higher ->)")
    axis.set_title("How the two halves differ", color=INK)
    axis.tick_params(left=False)

    figure.tight_layout()
    save(figure, "F4_two_islands" + MAP_SUFFIX)


# ------------------------------------------------------- F6 property maps
# One shared light-to-dark green ramp for all three panels. A single hue means the
# brightness alone carries the value, which stays readable in greyscale and for every
# kind of colour blindness.
PROPERTY_RAMP = mpl.colors.LinearSegmentedColormap.from_list(
    "palm", ["#f1f6f3", "#9ac6b2", "#2f9068", "#0b7350", "#04301f"]
)

# The properties to colour by, and how to label them. These are the transport features
# from Mayhew et al. 2022 -- how readily a molecule evaporates, how big it is, and how
# greasy it is -- which is what decides whether it can reach an olfactory receptor.
PROPERTIES = [
    ("Tb_joback", "boiling point (K)"),
    ("MolWt", "molecular weight (Da)"),
    ("LogP", "logP"),
]

# Molecules with no boiling-point estimate. Yellow shares no hue with the green ramp, so
# "no value" cannot be misread as "low value". It is also named in the legend, so the
# distinction never rests on colour alone.
NO_ESTIMATE_COLOUR = "#e8b21f"

# Contour lines marking where each odorant list sits. Kept as neutral dark annotation
# rather than a third colour scale, and distinguished by line style as well as shade so
# they stay separable in greyscale.
LEFFINGWELL_COLOUR = "#3f3f3c"


def density_contour(axis, x, y, extent, colour, style, label, share=0.75):
    """
    Outline the region containing most of a set of points.

    Bins the points, smooths the counts, and draws the single contour that encloses
    `share` of them. One line rather than several, because these sit on top of a
    colour scale that is already carrying information.

    Parameters
    ----------
    axis : matplotlib axis
    x, y : ndarray
        Coordinates of the set to outline.
    extent : tuple
        (xmin, xmax, ymin, ymax) of the plot, so the grid lines up with the map.
    colour, style, label : str
        Line colour, line style, and legend label.
    share : float, optional
        Fraction of the set to enclose.
    """
    from scipy.ndimage import gaussian_filter

    xmin, xmax, ymin, ymax = extent
    counts, _, _ = np.histogram2d(x, y, bins=180,
                                  range=[[xmin, xmax], [ymin, ymax]])
    smoothed = gaussian_filter(counts, sigma=2.5)

    # The density level below which `share` of the total mass sits.
    ordered = np.sort(smoothed.ravel())[::-1]
    cumulative = np.cumsum(ordered) / max(ordered.sum(), 1e-9)
    level = ordered[np.searchsorted(cumulative, share)]

    axis.contour(
        smoothed.T, levels=[level], colors=colour, linewidths=1.3, linestyles=style,
        extent=(xmin, xmax, ymin, ymax),
    )
    # Proxy artist, so the contour can appear in a legend.
    return mpl.lines.Line2D([], [], color=colour, lw=1.3, ls=style, label=label)


def figure6_property_maps(universe):
    """
    The map coloured by each transport feature, one standalone figure per feature.

    Writes F6a (boiling point), F6b (molecular weight) and F6c (logP). Each hexagon is
    the average value of the molecules falling inside it. Averaging in bins rather than
    drawing 726,000 overlapping dots means no molecule hides another, and the colour of
    a region is a real number rather than whatever happened to be painted last.

    Colour ranges are clipped at the 2nd and 98th percentile: without that, a handful
    of very large natural products stretch the scale so far that every odorant comes
    out the same shade.
    """
    coordinates = load_umap()
    extent = (coordinates[:, 0].min(), coordinates[:, 0].max(),
              coordinates[:, 1].min(), coordinates[:, 1].max())

    in_leffingwell_goodscents = (
        (universe.in_leffingwell + universe.in_goodscents) > 0
    ).values
    in_m2or = (universe.in_m2or == 1).values

    names = [n + MAP_SUFFIX for n in
             ["F6a_boiling_point", "F6b_molecular_weight", "F6c_logP"]]

    for name, (column, label) in zip(names, PROPERTIES):
        figure, axis = plt.subplots(figsize=(6.4, 5.6))

        values = pd.to_numeric(universe[column], errors="coerce").values.astype(float)
        low, high = np.nanpercentile(values, [2, 98])
        known = np.isfinite(values)

        # Yellow goes underneath. It shows through only where a bin holds no molecule
        # with an estimate, so yellow means "nothing to average here".
        if (~known).any():
            axis.hexbin(
                coordinates[~known, 0], coordinates[~known, 1],
                gridsize=110, mincnt=1, extent=extent,
                cmap=mpl.colors.ListedColormap([NO_ESTIMATE_COLOUR]),
                linewidths=0, rasterized=True,
            )

        bins = axis.hexbin(
            coordinates[known, 0], coordinates[known, 1], C=values[known],
            reduce_C_function=np.mean, gridsize=110, mincnt=3, extent=extent,
            cmap=PROPERTY_RAMP, vmin=low, vmax=high, linewidths=0, rasterized=True,
        )

        bar = figure.colorbar(bins, ax=axis, fraction=0.046, pad=0.02, extend="both")
        bar.set_label(label, fontsize=8.5, color=MUTED)
        bar.ax.tick_params(labelsize=7.5)
        bar.outline.set_visible(False)

        # Where each odorant list actually sits on the map.
        handles = [
            density_contour(axis, coordinates[in_leffingwell_goodscents, 0],
                            coordinates[in_leffingwell_goodscents, 1], extent,
                            LEFFINGWELL_COLOUR, "-",
                            f"Leffingwell / GoodScents "
                            f"(n={int(in_leffingwell_goodscents.sum()):,})"),
            density_contour(axis, coordinates[in_m2or, 0], coordinates[in_m2or, 1],
                            extent, M2OR_COLOUR, "--",
                            f"M2OR (n={int(in_m2or.sum()):,})"),
        ]
        if (~known).any():
            handles.append(mpl.patches.Patch(facecolor=NO_ESTIMATE_COLOUR,
                                             label="no estimate"))

        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_xlabel("UMAP 1")
        axis.set_ylabel("UMAP 2")

        n_missing = int((~known).sum())
        subtitle = (f"{n_missing:,} of {len(values):,} have no estimate"
                    if n_missing else f"all {len(values):,} molecules")
        axis.set_title(f"{label}\n{subtitle}", color=INK)
        axis.legend(handles=handles, frameon=False, fontsize=7,
                    loc="lower center", bbox_to_anchor=(0.5, -0.30),
                    labelcolor=INK, handlelength=2.0)

        figure.tight_layout()
        save(figure, name)


# ------------------------------------------------------- F7 where odorants come from
# Kingdoms in a fixed order, largest first, so the two panels line up and the reader
# does not have to re-find a taxon between them.
# Bacteria are taken at superkingdom rather than as NCBI's two kingdom-rank groups
# (Bacillati, Pseudomonadati): "bacteria" is the distinction that means something here,
# and splitting it reported the same result twice.
KINGDOM_ORDER = ["Viridiplantae", "Metazoa", "Fungi", "Bacteria"]

# Mammals appear as "source organisms" because a compound was measured in milk, breath
# or body odour, not because the animal makes it. They are drawn hatched rather than
# dropped, so the reader can see them and discount them.
HOST_TAXA = {"Primates", "Artiodactyla", "Mammalia", "Chordata", "Hominidae",
             "Bovidae", "Carnivora", "Rodentia", "Aves", "Homo sapiens"}

KINGDOM_LABEL = {
    "Viridiplantae": "green plants",
    "Metazoa": "animals",
    "Fungi": "fungi",
    "Bacteria": "bacteria",
    "plants": "green plants",
    "animals": "animals",
    "fungi": "fungi",
    "bacteria": "bacteria",
}

# What each taxon actually is, so a reader who does not know the orders can still read
# the figure. Kept short -- these are label suffixes, not definitions.
TAXON_EXAMPLES = {
    # orders
    "Lamiales": "mint, basil, lavender",
    "Apiales": "carrot, celery, cumin",
    "Sapindales": "citrus",
    "Myrtales": "eucalyptus, clove",
    "Brassicales": "mustard, cabbage",
    "Zingiberales": "ginger, banana",
    "Asparagales": "onion, garlic, vanilla",
    "Vitales": "grape",
    "Rosales": "rose, hops",
    "Solanales": "tomato, potato",
    "Poales": "grasses, cereals",
    "Laurales": "bay, cinnamon",
    "Piperales": "pepper",
    "Fabales": "legumes",
    "Malpighiales": "willow, flax",
    "Xylariales": "fungi",
    "Pleosporales": "fungi",
    "Hypocreales": "fungi",
    "Eurotiales": "moulds",
    "Agaricales": "mushrooms",
    "Oscillatoriales": "cyanobacteria",
    "Kitasatosporales": "Streptomyces",
    "Haplosclerida": "marine sponges",
    "Dictyoceratida": "marine sponges",
    "Malacalcyonacea": "soft corals",
    "Alcyonacea": "soft corals",
    # families
    "Lamiaceae": "mint, basil",
    "Apiaceae": "carrot, cumin",
    "Rutaceae": "citrus",
    "Myrtaceae": "eucalyptus, clove",
    "Zingiberaceae": "ginger, cardamom",
    "Amaryllidaceae": "onion, garlic",
    "Brassicaceae": "mustard, cabbage",
    "Caricaceae": "papaya",
    "Vitaceae": "grape",
    "Anacardiaceae": "mango, cashew",
    "Cannabaceae": "hops, cannabis",
    "Rosaceae": "apple, cherry",
    "Poaceae": "grasses",
    "Lauraceae": "bay, cinnamon",
    "Piperaceae": "pepper",
    "Asteraceae": "daisies, chamomile",
    "Aspergillaceae": "moulds",
    "Polyporaceae": "bracket fungi",
    "Celastraceae": "spindle trees",
    "Meliaceae": "mahogany, neem",
    "Rhodomelaceae": "red algae",
    "Alcyoniidae": "soft corals",
    "Streptomycetaceae": "Streptomyces",
}


def _kingdom_panel(axis, shares, present, panel_label, n_panel, n_other, title):
    """
    One horizontal paired-bar panel: the panel's source composition against the rest.

    `shares` is indexed by taxon and carries mean_share_panel, mean_share_background,
    fold and q. The comparison group is always non-odorant natural products -- odorants
    are removed from the background when the universe is built, so the two bars never
    share a molecule.
    """
    positions = np.arange(len(present))[::-1]
    panel_share = [100 * shares.mean_share_panel[k] for k in present]
    other_share = [100 * shares.mean_share_background[k] for k in present]

    axis.barh(positions + 0.2, panel_share, height=0.38, color=ODORANT_COLOUR,
              label=f"{panel_label} (n={n_panel:,})")
    axis.barh(positions - 0.2, other_share, height=0.38, color=BACKGROUND_COLOUR,
              label=f"other natural products (n={n_other:,})")

    for position, panel_value, other_value, taxon in zip(
            positions, panel_share, other_share, present):
        significance = "" if shares.q[taxon] < 0.05 else "  ns"
        axis.text(max(panel_value, other_value) + 1.6, position,
                  f"{panel_value:.1f}% vs {other_value:.1f}%   "
                  f"{shares.fold[taxon]:.2f}x{significance}",
                  va="center", fontsize=7, color=MUTED)

    axis.set_yticks(positions)
    axis.set_yticklabels([KINGDOM_LABEL.get(k, k) for k in present], fontsize=8.5)
    axis.set_xlim(0, 128)
    axis.set_xticks([0, 20, 40, 60, 80, 100])
    axis.set_xlabel("mean share of a molecule's source organisms (%)")
    axis.set_title(title, color=INK)
    axis.legend(frameon=False, fontsize=7.5, loc="lower right", labelcolor=INK)
    axis.tick_params(left=False)


def figure7_organism_kingdom(universe):
    """
    What kind of organism odorants are isolated from, for two odorant lists.

    Composition rather than presence/absence: of the organisms a molecule was isolated
    from, what share were plants, animals, fungi, bacteria. Odorants are recorded from
    far more organisms than other natural products, so a presence test would call every
    kingdom enriched at once.

    Both panels compare against non-odorant natural products. Odorants are taken out of
    the background when the universe is built, so nothing is on both sides of a bar.
    """
    all_file = RESULTS / "organism_rank_enrichment.csv"
    lg_file = RESULTS / "leffingwell_goodscents_kingdom_stats.csv"
    if not all_file.exists():
        print("s07 output missing; skipping F7", flush=True)
        return

    # Plants, animals and fungi come from the kingdom rows; bacteria from the
    # superkingdom row, which is the whole of Bacteria rather than one of its halves.
    ranks = pd.read_csv(all_file)
    everything = pd.concat([
        ranks.query("rank == 'kingdom' and taxon in @KINGDOM_ORDER"),
        ranks.query("rank == 'superkingdom' and taxon == 'Bacteria'"),
    ]).set_index("taxon")
    everything["mean_share_panel"] = everything.mean_share_odorant
    everything["fold"] = everything.share_ratio

    panels = [(everything, "all odorants", 1995, 267450,
               "A  All odorants")]

    if lg_file.exists():
        # s10 writes its own short labels; map them onto the NCBI names the rest of the
        # figure is keyed by, so both panels can share one row order.
        to_ncbi = {"plants": "Viridiplantae", "animals": "Metazoa",
                   "fungi": "Fungi", "bacteria": "Bacteria"}
        leffingwell = pd.read_csv(lg_file)
        leffingwell["taxon"] = leffingwell.kingdom.map(to_ncbi).fillna(leffingwell.kingdom)
        leffingwell = leffingwell.set_index("taxon")
        leffingwell["fold"] = leffingwell.fold_change
        panels.append((leffingwell, "Leffingwell / GoodScents",
                       int(leffingwell.n_panel.iloc[0]),
                       int(leffingwell.n_background.iloc[0]),
                       "B  Leffingwell / GoodScents only"))

    figure, axes = plt.subplots(1, len(panels), figsize=(7.4 * len(panels), 4.6),
                                squeeze=False)
    for axis, (shares, label, n_panel, n_other, title) in zip(axes[0], panels):
        present = [k for k in KINGDOM_ORDER if k in shares.index]
        _kingdom_panel(axis, shares, present, label, n_panel, n_other, title)

    figure.tight_layout()
    save(figure, "F7_organism_kingdom")


# ---------------------------------------------------- S1 the finer taxonomic ranks
def figureS1_organism_detail(universe, n_shown=12):
    """
    The same comparison at order and family level, as supplementary detail.

    Only taxa accounting for at least 0.5% of source organisms on one side are shown:
    below that the ratio of two near-zero shares is unstable and produces meaningless
    hundred-fold enrichments.
    """
    shares_file = RESULTS / "organism_rank_enrichment.csv"
    if not shares_file.exists():
        print("s07 outputs missing; skipping S1", flush=True)
        return

    table = pd.read_csv(shares_file)
    figure, axes = plt.subplots(1, 2, figsize=(12.6, 5.4))

    for axis, rank in zip(axes, ["order", "family"]):
        at_rank = table[(table["rank"] == rank) & (table.q < 0.05)].dropna(
            subset=["share_ratio"])
        # Mammals appear as "sources" because a compound was measured in milk, breath or
        # body odour rather than made there. Excluded outright rather than shown and
        # caveated, since they are not a biosynthetic origin at all.
        at_rank = at_rank[~at_rank.taxon.isin(HOST_TAXA)]
        substantial = at_rank[
            at_rank[["mean_share_odorant", "mean_share_background"]].max(axis=1) >= 0.005
        ].sort_values("share_ratio", ascending=False)

        half = n_shown // 2
        shown = pd.concat([substantial.head(half), substantial.tail(half)])
        positions = np.arange(len(shown))[::-1]
        log_ratio = np.log2(shown.share_ratio.values)
        colours = [ODORANT_COLOUR if v > 0 else BACKGROUND_COLOUR for v in log_ratio]

        axis.barh(positions, log_ratio, color=colours, height=0.66)

        axis.axvline(0, color=GRID, lw=1)
        for position, value, ratio in zip(positions, log_ratio, shown.share_ratio):
            offset = 0.14 if value >= 0 else -0.14
            axis.text(value + offset, position, f"{ratio:.2f}x", va="center",
                      ha="left" if value >= 0 else "right", fontsize=7, color=MUTED)

        limit = max(abs(log_ratio.min()), abs(log_ratio.max())) * 1.3
        axis.set_xlim(-limit, limit)
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [f"{taxon}\n{TAXON_EXAMPLES[taxon]}" if taxon in TAXON_EXAMPLES else taxon
             for taxon in shown.taxon], fontsize=7.5)
        axis.set_xlabel("log$_2$ ratio of source share  (odorant vs other)")
        axis.set_title(f"{rank.capitalize()}\ntop and bottom {half}, "
                       f"of {len(substantial)} tested", color=INK)
        axis.tick_params(left=False)

    figure.suptitle("Where odorants come from, at finer taxonomic ranks",
                    fontsize=11, color=INK, y=1.01)
    figure.text(0.5, -0.04,
                "Mammalian taxa are excluded: they appear as sources because a compound "
                "was measured in milk, breath or body odour, not synthesised there.",
                ha="center", fontsize=7.5, color=MUTED, style="italic")
    figure.tight_layout()
    save(figure, "S1_organism_order_family")


FIGURE_FUNCTIONS = {
    "0": figure0_pca,
    "1": figure1_map,
    "3": figure3_composition,
    "4": figure4_aromatic_split,
    "6": figure6_property_maps,
    "7": figure7_organism_kingdom,
    "s": figureS1_organism_detail,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default="01346",
                        help="which figures to draw, e.g. --only 13")
    parser.add_argument("--map", choices=["raw", "pca"], default="raw",
                        help="which UMAP to draw; pca writes _pca50 filenames")
    parser.add_argument("--results", default=None,
                        help="results directory to read (default ./results); lets the "
                             "same code draw either run")
    args = parser.parse_args()

    global MAP_CHOICE, MAP_SUFFIX
    MAP_CHOICE = args.map
    MAP_SUFFIX = "_pca50" if args.map == "pca" else ""

    if args.results:
        global RESULTS
        RESULTS = Path(args.results)
    print(f"reading {RESULTS}", flush=True)

    universe = load_universe()
    for key, draw in FIGURE_FUNCTIONS.items():
        if key in args.only:
            draw(universe)


if __name__ == "__main__":
    main()
