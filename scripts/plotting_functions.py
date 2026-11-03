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
        "font.family": "Liberation Sans",
        "pdf.fonttype": 42,          # embed as text, not outlines
        "ps.fonttype": 42,
        "svg.fonttype": "none",      
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