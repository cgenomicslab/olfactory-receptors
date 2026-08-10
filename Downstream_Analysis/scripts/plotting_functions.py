"""Shared matplotlib style for the downstream-analysis notebooks.

Usage from a notebook under notebooks/<subdir>/:

    import sys
    sys.path.append("../../scripts")
    from plotting_functions import custom_plots, FIGSIZE
    plt.rcParams.update(custom_plots())

Keeping the style here rather than inline means every figure in the analysis
shares one definition, and changing the look is a single edit.
"""

# Default figure size for single-panel figures, in inches.
FIGSIZE = (4, 3)


def custom_plots():
    """
    Create custom plotting style.

    Text is embedded as text rather than outlines (``fonttype`` 42, and
    ``svg.fonttype`` "none"), so labels stay selectable and editable when the
    exported SVG/PDF is opened in Illustrator.

    Returns
    -------
    updated_style : dict
        Dictionary with matplotlib parameters.
    """
    updated_style = {
        "font.family": "Liberation Sans",
        "pdf.fonttype": 42,          # embed as text, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",      # keep text editable in Illustrator
        "axes.linewidth": 0.8,
        "axes.edgecolor": "#c9c9c4",
        "text.color": "#8a8a86",
        "axes.labelcolor": "#c9c9c4",
        "xtick.color": "#8a8a86",
        "ytick.color": "#8a8a86",
        "savefig.dpi": 300,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
        "figure.dpi": 300,           # inline render resolution, not style
    }
    return updated_style
