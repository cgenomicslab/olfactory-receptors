#!/usr/bin/env python3
"""Bring axis typography in a figure SVG into line with the manuscript convention.

Axis labels and categorical tick names are set in black; numeric tick labels stay grey.
The rule is applied to every matplotlib axes in the file, including panels nested inside
a composite, so Extended Data figures end up matching the main figures and Extended Data
Fig. 1 without regenerating any panel.

Structure relied on (matplotlib's SVG output, stable across versions):

    <g id="matplotlib.axis_N">      the x or y axis of one panel
      <g id="xtick_N"> ...            tick mark and its label
      <g id="text_N"> ...             the axis label, last child of the axis group

so an axis label is a ``text_*`` group whose parent is a ``matplotlib.axis_*`` group, and a
tick label is a ``text_*`` group inside an ``xtick_*`` or ``ytick_*`` group. Nothing else is
touched: legends, titles, in-plot annotations and significance keys keep their colours.

    python scripts/restyle_axis_text.py FIG.svg [FIG2.svg ...] [--in-place]
"""

import argparse
import re
import shutil
from pathlib import Path

LABEL_COLOUR = "#000000"          # axis labels and categorical tick names
NUMBER_COLOUR = "#8a8a86"         # numeric tick labels

TAG = re.compile(r"<(/?)([A-Za-z_][\w.:-]*)([^>]*?)(/?)>", re.S)
ID = re.compile(r'\bid="([^"]*)"')
# Inside a text group every fill is a glyph fill, so both spellings can be rewritten
# wholesale. Plain labels carry one on the <text>; mathtext, which matplotlib draws as
# <tspan> runs inside a wrapping <g>, carries one on each.
FILL_IN_STYLE = re.compile(r"fill:\s*(?:#[0-9A-Fa-f]{3,8}|[A-Za-z]+)")
FILL_ATTR = re.compile(r'\bfill="(?:#[0-9A-Fa-f]{3,8}|[A-Za-z]+)"')
HAS_FILL = re.compile(r"\bfill[:=]")

# A tick label counts as a number if what is left after stripping the usual decorations is
# empty: digits, sign, decimal point, thousands separator, exponent, percent, and the
# multiplication sign matplotlib uses in offset text. Minus may be U+2212.
NUMERIC = re.compile(r"^[\s−+\-×0-9.,eE%^{}±×]*$")


def _entities(text):
    """Text content of an element, with the entities matplotlib escapes resolved."""
    bare = re.sub(r"<[^>]+>", "", text)
    for src, dst in (("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&#8722;", "−")):
        bare = bare.replace(src, dst)
    return " ".join(bare.split())


def _recolour(block, colour):
    """Recolour every glyph fill in one text group, and count the elements touched."""
    n = len(HAS_FILL.findall(block))
    block = FILL_IN_STYLE.sub(f"fill: {colour}", block)
    block = FILL_ATTR.sub(f'fill="{colour}"', block)
    if not n:                                    # a bare <text> inherits its colour
        block = block.replace("<text", f'<text fill="{colour}"', 1)
        n = 1
    return block, n


def restyle(svg):
    """Return the SVG with axis label and tick label colours normalised, plus a count."""
    spans = []                       # (start, end, colour) of every text group to recolour
    stack = []                       # open elements as (tag, id)
    for m in TAG.finditer(svg):
        closing, tag, attrs, selfclose = m.groups()
        if closing:
            if stack:
                stack.pop()
            continue
        ident = ID.search(attrs)
        ident = ident.group(1) if ident else ""
        if selfclose:
            continue
        if tag == "g" and ident.startswith("text_") and stack:
            parent = stack[-1][1]
            if parent.startswith("matplotlib.axis_"):
                spans.append([m.start(), None, LABEL_COLOUR])
            elif parent.startswith(("xtick_", "ytick_")):
                spans.append([m.start(), None, None])   # colour decided from the content
            stack.append((tag, ident))
            continue
        stack.append((tag, ident))

    # Close each recorded group by walking forward to its matching </g>.
    for span in spans:
        depth, pos = 1, svg.index(">", span[0]) + 1
        while depth:
            nxt = re.search(r"<g\b[^>]*?(/?)>|</g>", svg[pos:])
            if not nxt:
                break
            if nxt.group(0) == "</g>":
                depth -= 1
            elif not nxt.group(1):
                depth += 1
            pos += nxt.end()
        span[1] = pos

    changed, out, cursor = 0, [], 0
    for start, end, colour in sorted(spans, key=lambda s: s[0]):
        if start < cursor:                       # nested groups: keep the outer decision
            continue
        block = svg[start:end]
        if colour is None:
            colour = NUMBER_COLOUR if NUMERIC.match(_entities(block)) else LABEL_COLOUR
        new, n = _recolour(block, colour)
        changed += n
        out.append(svg[cursor:start])
        out.append(new)
        cursor = end
    out.append(svg[cursor:])
    return "".join(out), changed


# --------------------------------------------------------------- type size in composites
# A panel placed in an Extended Data composite is drawn at whatever scale makes the page
# work, and a panel scaled to 45% carries 9 pt type down to 4 pt. Rather than re-render the
# panel, each text element is scaled about its own anchor point, which enlarges the type
# without moving where it sits or disturbing the glyph spacing matplotlib computed for
# mathtext. Only text inside a matplotlib text group is touched.
TEXT_G = re.compile(r'<g id="text_\d+">')
LEGEND_G = re.compile(r'<g id="legend_\d+">')
NUM_XY = re.compile(r'\b(?:x|y)="([-\d.eE]+)"')
PATH_D = re.compile(r'\bd="([^"]+)"')
XY = re.compile(r'\bx="([-\d.eE]+)"\s+y="([-\d.eE]+)"')
TRANSLATE_G = re.compile(r'(<g\b[^>]*?transform=")(translate\([^)]*\))(")')


def _scale_about(match, k):
    """Append a scale about the element's own anchor to a <text> transform."""
    attrs = match.group(1)
    xy = XY.search(attrs)
    if not xy:
        return match.group(0)
    x, y = xy.group(1), xy.group(2)
    extra = f"translate({x} {y}) scale({k}) translate(-{x} -{y})"
    if 'transform="' in attrs:
        attrs = attrs.replace('transform="', f'transform="{extra} ', 1)
    else:
        attrs = attrs.rstrip() + f' transform="{extra}"'
    return f"<text{attrs}>"


def _close(svg, start):
    """End offset of the <g> opening at `start`, matching by depth."""
    depth, pos = 1, svg.index(">", start) + 1
    while depth:
        nxt = re.search(r"<g\b[^>]*?(/?)>|</g>", svg[pos:])
        if not nxt:
            break
        if nxt.group(0) == "</g>":
            depth -= 1
        elif not nxt.group(1):
            depth += 1
        pos += nxt.end()
    return pos


def _bbox(block):
    """Rough bounding box of a group, from the coordinates it states explicitly."""
    xs, ys = [], []
    for m in re.finditer(r'\bx="([-\d.eE]+)"\s+y="([-\d.eE]+)"', block):
        xs.append(float(m.group(1))); ys.append(float(m.group(2)))
    for d in PATH_D.findall(block):
        nums = [float(v) for v in re.findall(r"[-\d.]+(?:[eE][-+]?\d+)?", d)]
        xs += nums[0::2]; ys += nums[1::2]
    if not xs or not ys:
        return None
    return (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2


def scale_legends(svg, k):
    """Scale each legend as a whole about its centre, frame and swatches included.

    A legend's text cannot be enlarged on its own: matplotlib has already laid the entries
    out against a fixed frame, so bigger type would run into the next swatch. Scaling the
    whole group keeps that layout intact and simply makes the legend larger.
    """
    if k == 1:
        return svg, 0
    out, cursor, n = [], 0, 0
    for m in LEGEND_G.finditer(svg):
        if m.start() < cursor:
            continue
        end = _close(svg, m.start())
        block = svg[m.start():end]
        centre = _bbox(block)
        if centre is None:
            continue
        cx, cy = centre
        opening = block[:block.index(">") + 1]
        moved = opening[:-1] + (f' transform="translate({cx:.2f} {cy:.2f}) scale({k}) '
                                f'translate({-cx:.2f} {-cy:.2f})">')
        out.append(svg[cursor:m.start()])
        out.append(moved + block[len(opening):])
        cursor = end
        n += 1
    out.append(svg[cursor:])
    return "".join(out), n


ROT90 = re.compile(r'transform="([^"]*?)rotate\(-90 ([-\d.eE]+) ([-\d.eE]+)\)')


def _nudge_outer_labels(svg, k):
    """Move a right-hand axis label clear of tick labels that have just grown.

    A colour bar puts its tick labels to the right of the bar, anchored at their left, so
    enlarging them pushes them towards the axis label. The label is moved out by the same
    amount rather than left to collide.
    """
    out, cursor = [], 0
    for m in re.finditer(r'<g id="matplotlib\.axis_\d+">', svg):
        if m.start() < cursor:
            continue
        end = _close(svg, m.start())
        block = svg[m.start():end]
        ticks = [block[t.start():_close(block, t.start())]
                 for t in re.finditer(r'<g id="ytick_\d+">', block)]
        if not ticks or 'text-anchor: start' not in "".join(ticks):
            continue
        widest = 0.0
        for t in ticks:
            body = _entities(t)
            size = re.search(r"font-size:\s*([\d.]+)", t)
            widest = max(widest, len(body) * 0.55 * (float(size.group(1)) if size else 10.0))
        dx = (k - 1) * widest
        label = block.rfind('<g id="text_')
        if label < 0:
            continue
        head, tail = block[:label], block[label:]
        tail = ROT90.sub(lambda t: 'transform="%stranslate(%.2f 0) rotate(-90 %s %s)'
                         % (t.group(1), dx, t.group(2), t.group(3)), tail, count=1)
        out.append(svg[cursor:m.start()]); out.append(head + tail)
        cursor = end
    out.append(svg[cursor:])
    return "".join(out)


def _in_group(svg, pos, prefix):
    """True if the element at `pos` sits inside an open group whose id starts with prefix."""
    depth = 0
    tags = list(re.finditer(r'<g\b([^>]*?)(/?)>|</g>', svg[:pos]))
    for m in reversed(tags):
        if m.group(0) == "</g>":
            depth += 1
        elif not m.group(2):
            if depth:
                depth -= 1
            else:
                ident = ID.search(m.group(1) or "")
                if ident and ident.group(1).startswith(prefix):
                    return True
    return False


def _shift_scale(match, k, shift):
    """Rewrite translate(a b) as translate(a-shift b) scale(k)."""
    inside = match.group(2)[match.group(2).index("(") + 1:match.group(2).rindex(")")]
    nums = [float(v) for v in re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", inside)]
    a, b = (nums + [0.0, 0.0])[:2]
    return f"{match.group(1)}translate({a - shift:.3f} {b:.3f}) scale({k}){match.group(3)}"


def scale_text(svg, k, legend_k=None):
    """Enlarge axis and annotation labels by k; legends are scaled whole, at legend_k."""
    if k == 1:
        return svg, 0
    legends = [(m.start(), _close(svg, m.start())) for m in LEGEND_G.finditer(svg)]
    out, cursor, n = [], 0, 0
    for m in TEXT_G.finditer(svg):
        if m.start() < cursor:
            continue
        if any(a < m.start() < b for a, b in legends):    # handled by scale_legends
            continue
        pos = _close(svg, m.start())
        block = svg[m.start():pos]
        # mathtext: matplotlib wraps positioned <tspan> glyphs in a translated <g> and has
        # already placed the string, so the group is scaled as a whole about the end it is
        # aligned to -- the right end for a y tick label, the centre for an x tick label --
        # rather than about each glyph, which would break the spacing.
        if "<tspan" in block:
            width = max([float(v) for v in re.findall(r'<tspan x="([-\d.eE]+)"', block)] or [0])
            align = 1.0 if _in_group(svg, m.start(), "ytick_") else (
                    0.5 if _in_group(svg, m.start(), "xtick_") else 0.0)
            shift = (k - 1) * width * align
            block = TRANSLATE_G.sub(
                lambda t: _shift_scale(t, k, shift), block, count=1)
        else:
            block = re.sub(r"<text\b([^>]*)>", lambda t: _scale_about(t, k), block)
        n += block.count("scale(")
        out.append(svg[cursor:m.start()])
        out.append(block)
        cursor = pos
    out.append(svg[cursor:])
    svg = _nudge_outer_labels("".join(out), k)
    svg, _ = scale_legends(svg, legend_k if legend_k is not None else 1 + (k - 1) * 0.55)
    return svg, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("svg", nargs="+", type=Path)
    ap.add_argument("--in-place", action="store_true",
                    help="overwrite the input after keeping a .bak copy")
    ap.add_argument("--suffix", default=".restyled",
                    help="written as NAME<suffix>.svg when not in place")
    args = ap.parse_args()

    for path in args.svg:
        text = path.read_text(encoding="utf-8", errors="replace")
        new, n = restyle(text)
        if args.in_place:
            shutil.copyfile(path, path.with_suffix(path.suffix + ".bak"))
            path.write_text(new, encoding="utf-8")
            dest = path
        else:
            dest = path.with_name(path.stem + args.suffix + path.suffix)
            dest.write_text(new, encoding="utf-8")
        print(f"{path.name}: {n} axis text elements recoloured -> {dest.name}")


if __name__ == "__main__":
    main()
