# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as py
from matplotlib import colors as matcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot(config, fit, fig_file, dofs=None):
    """
    Computes and displays the correlation coefficients
    between parameters in a heat map

    Parameters
    ----------
        config : dict
            configuration dictionary
        fit : dict
            posterior distributions dictionary
        fig_file : str
            fit name
        dofs: dict
            dependent degrees of freedom to keep or hide

    """
    if dofs is None:
        dofs = {"show": [], "hide": []}

    coeff_list = list(fit.keys())
    param_data = pd.DataFrame(fit.values()).T
    correlations = param_data.corr()
    rows_to_keep = []
    for i, _ in enumerate(correlations):
        for j, _ in enumerate(correlations[i]):
            if coeff_list[i] not in  config["coefficients"]:
                continue
            if (coeff_list[i] not in dofs["show"]) and config["coefficients"][
                coeff_list[i]
            ]["fixed"] is not False:
                continue
            if coeff_list[i] in dofs["hide"]:
                continue
            if i != j and (correlations[i][j] > 0.5 or correlations[i][j] < -0.5):
                if i not in rows_to_keep:
                    # Uncommnent to avoid displaying uncorrelated coeffs after a rotation
                    # if config["coefficients"][coeff_list[j]]["fixed"] is False \
                    #     and coeff_list[j] not in dofs["hide"]:
                    #     rows_to_keep.append(i)
                    # if j not in rows_to_keep:
                    #    rows_to_keep.append(j)
                    rows_to_keep.append(i)


    correlations = np.array(correlations)[np.array(rows_to_keep), :]
    correlations = np.array(correlations)[:, np.array(rows_to_keep)]

    labels = np.array(coeff_list)[np.array(rows_to_keep)]
    npar = len(labels)

    fig = py.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    colors = py.rcParams["axes.prop_cycle"].by_key()["color"]
    cmap = matcolors.ListedColormap([colors[0], "lightgrey", colors[1]])
    norm = matcolors.BoundaryNorm([-1.0, -0.5, 0.5, 1.0], cmap.N)

    divider = make_axes_locatable(ax)
    cax = ax.matshow(correlations, cmap=cmap, norm=norm)
    fig.colorbar(cax, cax=divider.append_axes("right", size="5%", pad=0.1))

    for i in range(npar):
        for j in range(npar):
            c = correlations[j, i]
            if c >= 0.1 or c <= -0.1:
                ax.text(
                    i, j, str(np.round(c, 1)), va="center", ha="center", fontsize=10
                )
    ticks = np.arange(0, npar, 1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="x", rotation=90, labelsize=15)
    ax.tick_params(axis="y", labelsize=15)
    py.tight_layout()
    py.savefig(fig_file)
