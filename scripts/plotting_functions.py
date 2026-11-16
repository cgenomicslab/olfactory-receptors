"""Shared matplotlib style for every notebook in the repository.

    import sys
    sys.path.append("../scripts")        
    from plotting_functions import custom_plots, FIGSIZE
    plt.rcParams.update(custom_plots())

"""

FIGSIZE = (4, 3)

GREY = "#595959"

def custom_plots():
    """Return the shared matplotlib parameters, as a dict for ``rcParams.update``.

    """
    return {
        # Arial and Helvetica have the same letter widths as Liberation Sans, so an SVG opened where
        # Liberation Sans is not installed (a browser, GitHub) keeps its layout.
        "font.family": "sans-serif",
        "font.sans-serif": ["Liberation Sans", "Arial", "Helvetica", "DejaVu Sans"],
        "pdf.fonttype": 42,          # embed as text, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",      
        # Text sizes in points. Every figure is 4 x 3 in and is placed without scaling.
        "font.size": 7,              # annotations and any text with no size of its own
        "axes.titlesize": 8,
        "axes.titleweight": "normal",
        "axes.titlecolor": GREY,
        "figure.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "legend.frameon": False,
        # Lines
        "lines.linewidth": 1.5,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "figure.figsize": FIGSIZE,
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
        "figure.dpi": 200,
    }