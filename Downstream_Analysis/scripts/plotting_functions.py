def custom_plots():
    """
    Create custom plotting style.

    Returns
    -------
    updated_style : dict
        Dictionary with matplotlib parameters.
    """
    fsize = 12
    updated_style = {
        "text.usetex": False,
        "font.family": "Serif",
        "font.weight": "bold",
        "axes.labelsize": fsize,
        "axes.titlesize": fsize,
        "font.size": fsize,
        "grid.color": "black",
        "grid.linewidth": 0.2,
        "legend.fontsize": fsize-2,
        "xtick.labelsize": fsize,
        "ytick.labelsize": fsize,
        "axes.linewidth": 2.5,
        "lines.markersize": 5.0,
        "lines.linewidth": 2.0,
        "xtick.major.width": 2.2,
        "ytick.major.width": 2.2,
        "axes.edgecolor": "black",
        "axes.labelweight": "bold",
        "axes.titleweight": "bold",
        "axes.spines.right": False,
        "axes.spines.top": False,
        "svg.fonttype": "none"
    }
    return updated_style