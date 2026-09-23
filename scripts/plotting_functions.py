"""Shared matplotlib style for every notebook in the repository.

    import sys
    sys.path.append("../scripts")        # depth varies; see the notebook
    from plotting_functions import custom_plots, FIGSIZE
    plt.rcParams.update(custom_plots())

One definition, so changing the look of every figure is a single edit.
"""

# Default figure size for single-panel figures, in inches.
FIGSIZE = (4, 3)

# Axes, labels and ticks are all the same grey.
GREY = "#595959"


def custom_plots():
    """Return the shared matplotlib parameters, as a dict for ``rcParams.update``.

    Text is embedded as text rather than outlines (``fonttype`` 42, and
    ``svg.fonttype`` "none"), so labels stay selectable and editable when the
    exported SVG or PDF is opened in Illustrator.
    """
    return {
        "font.family": "Liberation Sans",
        "pdf.fonttype": 42,          # embed as text, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",      # keep text editable in Illustrator
        "axes.linewidth": 0.8,
        "axes.edgecolor": GREY,
        "text.color": GREY,
        "axes.labelcolor": GREY,
        "xtick.color": GREY,
        "ytick.color": GREY,
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        # Inline render resolution only -- what a notebook stores for display.
        # Exported figures are governed by savefig.dpi above, and are vector in
        # SVG/PDF regardless. At 300 the stored PNGs pushed the merged notebooks
        # past 10 MB, which is where GitHub stops rendering a notebook and shows
        # a download link instead.
        "figure.dpi": 110,
    }
