#!/usr/bin/env python3
"""Assemble Extended Data figure drafts from the per-notebook panel SVGs.

Each Extended Data figure is a single A4-portrait SVG in which every source panel is
nested as its own <svg> element, so the output stays fully vector and every panel can
still be moved or restyled in Inkscape. Panel letters are drawn in Arial bold to match
the main figures.

Paths are repository-relative; the output directory is passed on the command line
because the manuscript lives outside the code repository.

    python scripts/build_extended_figures.py --out <manuscript>/02.Figures/Final
"""

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from restyle_axis_text import restyle, scale_text

REPO = Path(__file__).resolve().parent.parent
REF = REPO / "Downstream_Analysis/notebooks/reference_tree/Figures"
ANC = REPO / "Downstream_Analysis/notebooks/ancestral/Figures"

PAGE_W, PAGE_H = 595.28, 841.89          # A4 portrait, points
MARGIN = 24.0
GUTTER = 12.0
LETTER_H = 13.0                          # space reserved above each panel for its letter

# Each figure is a list of rows; each row is a list of source SVGs that share the row
# width in proportion to their natural widths.
FIGURES = {
    "ext_figure_2": {
        "title": "Model calibration and error structure",
        "rows": [
            [REF / "threshold_calibration.svg"],
            [REF / "confusion_matrix_common_pairs.svg",
             REF / "probability_density_by_confusion_matrix_category.svg"],
            [REF / "proportions_scatter.svg"],
        ],
    },
    "ext_figure_3": {
        "title": "The activation matrix in phylogenetic order",
        # The single panel is the phylogeny beside the matrix, assembled by
        # build_tree_heatmap_panel() into TREE_HEATMAP_PANEL before this figure is laid out.
        "rows": [[None]],
        "letters": False,
        "text_scale": 1.15,
    },
    "ext_figure_4": {
        "title": "Ordination and chemistry of the activation clusters",
        # Three panels on one page place them near half size, which would carry their
        # 9-10.5 pt type down to about 5 pt; the labels are enlarged back into the band the
        # main figures use.
        "text_scale": 1.30,
        "rows": [
            [REF / "pcoa_hybrid_density.svg",
             REF / "chemspace_clusters_hybrid_density_groups.svg"],
            [REF / "chemspace_clusters_hybrid_density_effects.svg"],
        ],
    },
    "ext_figure_5": {
        "title": "Ancestral repertoire sizes and functional turnover along the tree",
        "rows": [
            [ANC / "03_ancestor_vs_extant.svg", ANC / "01_repertoire_turnover.svg"],
            [ANC / "03_ligands_vs_root_distance.svg"],
            [ANC / "04_similarity_vs_branch.svg"],
        ],
    },
    "ext_figure_6": {
        "title": "Chemistry and geometry of the two founding duplications",
        "rows": [
            [ANC / "05_gain_loss.svg"],
            [ANC / "05_functional_groups.svg"],
            [ANC / "05_optD_shift_vectors.svg", ANC / "05_orthogonality_null.svg"],
            [ANC / "05_optE_pca_chemspace.svg"],
        ],
    },
}


# Panel titles to remove when a notebook figure is placed in an Extended Data composite.
#
# Nature figures do not carry per-panel titles: the description, and any statistic shown in
# a title, belongs in the legend. Extended Data is not copy-edited by the journal, so a title
# left here would publish as-is. The notebook figures keep their titles -- they are useful
# when reading the analysis -- and only the composite is stripped. Every statistic removed
# below is carried into the Extended Data legend instead.
#
# Axis labels, legend entries, significance keys and in-plot data annotations are NOT listed
# and are left untouched.
PANEL_TITLES = {
    "threshold_calibration.svg": [
        "Calibration: measured proportion 4.74%",
        "Discrimination agrees with calibration"],
    "proportions_scatter.svg": [
        "Binding proportion per receptor", "(one point = one human OR)",
        "Pearson r = 0.912 Spearman \u03c1 = 0.403"],
    "pcoa_hybrid_density.svg": [
        "PCoA of odorant co-tuning (Jaccard) \u2014 hybrid_density",
        "non-Euclidean (neg. eigenvalue mass 9%)"],
    "chemspace_clusters_hybrid_density_groups.svg": [
        "Functional groups by activation cluster \u2014 hybrid_density"],
    "chemspace_clusters_hybrid_density_effects.svg": [
        "What each activation cluster is made of \u2014 hybrid_density"],
    "03_ancestor_vs_extant.svg": ["median 7 vs 4 MWU p = 1.8e-08"],
    "01_repertoire_turnover.svg": ["Repertoire turnover from the common ancestor"],
    "03_ligands_vs_root_distance.svg": [
        "Pearson r = +0.058, p = 0.088 \u00b7 slope = +6.1 ligands per unit distance"],
    "04_similarity_vs_branch.svg": [
        "Spearman r = -0.455, p = 3.6e-45", "short 0.67 mid 0.50 long 0.28"],
    "04_threshold_check.svg": [
        "threshold-free check r = -0.582", "flat 0.3-0.8; density cut half-life 0.162"],
    "05_gain_loss.svg": ["Turnover along each edge"],
    "05_functional_groups.svg": ["Functional-group prevalence per node"],
    "05_optD_shift_vectors.svg": [
        "shift in 13-descriptor space, from each parent to what it gained"],
    "05_orthogonality_null.svg": ["0 of 2000 random draws come as close to 90\u00b0"],
    "05_optE_pca_chemspace.svg": [
        "the two duplications moved into different regions of odorant chemical space"],
    "heatmap_hybrid_density_tree_order.svg": [
        "Hybrid landscape (M2OR truth + density-calibrated fill)",
        "experimental where available, else top-N by rank"],
}


def strip_titles(body, filename):
    """Delete the <g id="text_N"> blocks holding this panel's title text."""
    wanted = PANEL_TITLES.get(filename)
    if not wanted:
        return body
    # Matched on prefix, not equality: matplotlib emits a multi-line title as a single
    # <g> block, sometimes with a significance key appended, and the entities inside are
    # escaped. The prefix is distinctive enough to identify the block unambiguously.
    wanted = tuple(" ".join(w.split()) for w in wanted)
    out, cursor = [], 0
    for match in re.finditer(r'<g id="text_\d+">', body):
        if match.start() < cursor:
            continue
        depth, pos = 1, match.end()
        while depth:                                   # match the closing tag by depth
            nxt = re.search(r'<g\b[^>]*>|</g>', body[pos:])
            if not nxt:
                break
            depth += 1 if nxt.group(0) != "</g>" else -1
            pos += nxt.end()
        block = body[match.start():pos]
        text = " ".join(re.sub(r"<[^>]+>", "", block).split())
        if text.startswith(wanted):
            out.append(body[cursor:match.start()])
            cursor = pos
    out.append(body[cursor:])
    return "".join(out)


def source_size(path):
    """Natural width and height of an SVG, preferring viewBox over width/height."""
    head = path.read_text(encoding="utf-8", errors="replace")[:2000]
    vb = re.search(r'viewBox="\s*([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)\s+([-\d.eE]+)', head)
    if vb:
        return float(vb.group(3)), float(vb.group(4))
    w = re.search(r'width="([\d.]+)', head)
    h = re.search(r'height="([\d.]+)', head)
    if not (w and h):
        raise ValueError(f"cannot determine size of {path}")
    return float(w.group(1)), float(h.group(1))


def inner_svg(path, x, y, w, h):
    """Nest a source SVG at (x, y) scaled to fit (w, h), preserving aspect ratio."""
    body = path.read_text(encoding="utf-8", errors="replace")
    body = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", body)
    body = re.sub(r"<!DOCTYPE[^>]*>\s*", "", body)
    body = re.sub(r"<!--.*?-->", "", body, flags=re.S)
    # Strip the outer <svg ...> wrapper and keep its viewBox for the nested element.
    m = re.search(r"<svg\b([^>]*)>", body, flags=re.S)
    attrs, inner = m.group(1), body[m.end():]
    inner = inner[:inner.rfind("</svg>")]
    inner = strip_titles(inner, path.name)
    vb = re.search(r'viewBox="([^"]+)"', attrs)
    if vb:
        view = vb.group(1)
    else:
        sw, sh = source_size(path)
        view = f"0 0 {sw} {sh}"
    return (
        f'<svg x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" '
        f'viewBox="{view}" preserveAspectRatio="xMidYMid meet" overflow="visible">'
        f"{inner}</svg>"
    )


# ---------------------------------------------------------------- tree + heatmap
# The activation heatmap in Fig. 2a is ordered by co-tuning: receptor rows are placed by
# average-linkage clustering of their Jaccard distances, not by the phylogeny. The panel
# built here is the same matrix with rows in phylogenetic order, drawn beside the tree that
# defines that order, so the two can be read against each other. It is panel a of Extended
# Data Fig. 3.
# iTOL rendering of Phylogenetic_Analysis/data/HumanTree/human433_MFP_reorder.tree.
# Committed rather than rebuilt: iTOL is a web service, so the drawing cannot be
# regenerated by a script, and the tip order in this file is what the heatmap rows
# are aligned against.
TREE_SVG = REPO / "Phylogenetic_Analysis/figures/human433_MFP_reorder.tree.itol.svg"
HEATMAP_TREE_ORDER = REF / "heatmap_hybrid_density_tree_order.svg"
TREE_HEATMAP_PANEL = "_panel_tree_heatmap.svg"   # intermediate, built into Ext. Fig. 3

# Bounding box of the iTOL drawing itself, from `inkscape --query-id=treeHolder`. The file
# declares a 1920x932 canvas but the tree occupies only this region of it.
TREE_BOX = (816.422, 314.949, 78.3443, 338.101)          # x, y, w, h

# The heatmap's data area, taken from the <image> element matplotlib emits for the
# rasterised imshow. The 433 receptor rows span exactly this y range in the file's own
# units, which is what the tree tips have to line up with.
HEAT_DATA_Y0, HEAT_DATA_H = 98.88, 362.4
HEAT_W, HEAT_H = 804.528, 506.784

# iTOL draws at stroke-width 4 inside a group scaled by 0.02608, so branches land at about
# 0.07 pt once the tree is sized to the heatmap -- invisible in print. Scaled up to sit
# near 0.4 pt.
TREE_STROKE_SCALE = 8


def build_tree_heatmap_panel(out_dir, gap=8.0):
    """Write the phylogeny beside the phylogenetically ordered activation matrix.

    Emitted as a self-contained panel sized to its own content, not to the page, so it can
    be placed in an Extended Data composite like any other source SVG. The tree is scaled
    so that its tip span equals the height of the heatmap's data area, which is what makes
    row *i* of the matrix line up with tip *i* of the tree.
    """
    tree_over_heat = HEAT_DATA_H / TREE_BOX[3]
    tree_w = TREE_BOX[2] * tree_over_heat
    panel_w, panel_h = tree_w + gap + HEAT_W, HEAT_H

    tree = TREE_SVG.read_text(encoding="utf-8", errors="replace")
    tree = re.sub(r"^\s*<\?xml[^>]*\?>\s*", "", tree)
    tree = tree[re.search(r"<svg\b[^>]*>", tree).end():]
    tree = tree[:tree.rfind("</svg>")]
    tree = tree.replace('stroke-width="4"', f'stroke-width="{4 * TREE_STROKE_SCALE}"')

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{panel_w:.2f}pt" height="{panel_h:.2f}pt" '
        f'viewBox="0 0 {panel_w:.2f} {panel_h:.2f}">',
        "<title>Activation matrix in phylogenetic order</title>",
        f'<svg x="0" y="{HEAT_DATA_Y0:.2f}" width="{tree_w:.2f}" height="{HEAT_DATA_H:.2f}" '
        f'viewBox="{TREE_BOX[0]} {TREE_BOX[1]} {TREE_BOX[2]} {TREE_BOX[3]}" '
        f'preserveAspectRatio="none">{tree}</svg>',
        inner_svg(HEATMAP_TREE_ORDER, tree_w + gap, 0, HEAT_W, HEAT_H),
        "</svg>",
    ]
    out = out_dir / TREE_HEATMAP_PANEL
    out.write_text("\n".join(parts), encoding="utf-8")
    print(f"{out.name}  tree {tree_w:.0f} pt beside heatmap {HEAT_W:.0f} pt "
          f"(panel {panel_w:.0f}x{panel_h:.0f} pt)")
    return out


def build(name, spec, out_dir, version="v2"):
    """Lay panels out at one common scale, so type size is consistent across panels."""
    content_w = PAGE_W - 2 * MARGIN
    rows = [[out_dir / TREE_HEATMAP_PANEL if p is None else p for p in row]
            for row in spec["rows"]]
    rows = [[(p, *source_size(p)) for p in row] for row in rows]

    # One scale for the whole page: the widest row must fit the content width, and the
    # whole stack must fit the page height. Panels are never stretched independently.
    widest = max(sum(w for _, w, _ in row) + GUTTER * (len(row) - 1) for row in rows)
    stack = sum(max(h for _, _, h in row) + LETTER_H + GUTTER for row in rows) - GUTTER
    # Enlarged labels overflow the panel boxes they were laid out in, so a page that uses
    # them keeps a little more room at the foot than the panel geometry alone asks for.
    slack = 16.0 if spec.get("text_scale", 1.0) != 1.0 else 0.0
    scale = min(content_w / widest, (PAGE_H - 2 * MARGIN - slack) / stack)

    placed, y = [], MARGIN
    for row in rows:
        row_w = (sum(w for _, w, _ in row) + GUTTER * (len(row) - 1)) * scale
        x = MARGIN + (content_w - row_w) / 2          # centre each row on the page
        row_h = max(h for _, _, h in row) * scale
        for src, w, h in row:
            placed.append((src, x, y + LETTER_H, w * scale, h * scale))
            x += w * scale + GUTTER
        y += LETTER_H + row_h + GUTTER

    # A figure whose panels do not fill the page is trimmed to its content rather than
    # left with a band of white below the last panel. It still fits one page, which is all
    # the journal asks of an Extended Data item.
    page_h = min(PAGE_H, y - GUTTER + MARGIN + slack)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
        f'width="{PAGE_W}pt" height="{page_h:.2f}pt" viewBox="0 0 {PAGE_W} {page_h:.2f}">',
        f'<title>{spec["title"]}</title>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]
    for i, (src, x, yy, w, h) in enumerate(placed):
        if len(placed) > 1:                      # a one-panel figure carries no panel letter
            parts.append(
                f'<text x="{x:.2f}" y="{yy - 3:.2f}" font-family="Arial, Helvetica, sans-serif" '
                f'font-size="11" font-weight="bold" fill="#000000">'
                f'{chr(ord("a") + i)}</text>'
            )
        parts.append(inner_svg(src, x, yy, w, h))
    parts.append("</svg>")

    svg, enlarged = scale_text("\n".join(parts), spec.get("text_scale", 1.0))
    svg, recoloured = restyle(svg)
    out = out_dir / f"{name}.{version}.svg"
    out.write_text(svg, encoding="utf-8")
    panels = " ".join(chr(ord("a") + i) + "=" + s.name for i, (s, *_) in enumerate(placed))
    note = f", type x{spec['text_scale']}" if spec.get("text_scale", 1.0) != 1.0 else ""
    print(f"{out.name}  ({len(placed)} panels, scale {scale:.2f}, page "
          f"{PAGE_W:.0f}x{page_h:.0f} pt, {recoloured} labels restyled{note})\n    {panels}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", nargs="*", default=sorted(FIGURES))
    ap.add_argument("--version", default="v2", help="version tag in the output filename")
    ap.add_argument("--keep-intermediate", action="store_true",
                    help="leave the phylogeny-plus-matrix panel on disk after the build")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    panel = build_tree_heatmap_panel(args.out)
    for name in args.only:
        build(name, FIGURES[name], args.out, args.version)
    if not args.keep_intermediate:
        panel.unlink()
