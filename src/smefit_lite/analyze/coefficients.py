# # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as py
from matplotlib import cm
import rich


from .. import coefficients_utils as utils


class CoefficientsPlotter:
    """
    Plots central values + 95% CL errors, 95% CL bounds and residuals.

    Takes into account parameter constraints and displays
    all non-zero parameters.

    Parameters
    ----------
        config : dict
            configuration dictionary
        report_folder : str
            report folder, where plots are saved
        hid_dofs: list
            list of coefficients to hide
    """

    def __init__(self, config, report_folder, hide_dofs=None):

        hide_dofs = hide_dofs or []
        self.report_folder = report_folder

        coeff_config = utils.coeff_by_group()
        # coeff list contains the coefficents that are fitted
        # in at least one fit included in the report
        self.coeff_list = []
        for group in coeff_config.values():
            for c, _ in group:
                if c in hide_dofs:
                    continue
                if np.all(
                    [
                        config[k]["coefficients"][c]["fixed"] is True
                        for k in config
                        if c in config[k]["coefficients"]
                    ]
                ):
                    continue
                if c not in self.coeff_list and np.any(
                    [c in config[k]["coefficients"] for k in config]
                ):
                    self.coeff_list.append(c)

        self.npar = len(self.coeff_list)
        self.coeff_labels = [utils.latex_coeff(name) for name in self.coeff_list]

    def plot_coeffs(self, bounds):
        """
        Plot central value + 95% CL errors

        Parameters
        ----------
            bounds: dict
                confidence level bounds per fit and coefficient
                Note: double solutions are appended under "2"
        """

        py.figure(figsize=(12, 6))
        ax = py.subplot(111)

        # X-axis
        X = 2 * np.array(range(self.npar))
        # Spacing between fit results
        val = np.linspace(-0.1 * len(bounds), 0.1 * len(bounds), len(bounds))
        colors = cm.get_cmap("tab20")
        # loop over fits
        for i, name in enumerate(bounds):
            for cnt, coeff in enumerate(self.coeff_list):
                if cnt == 0:
                    label = name
                if coeff not in bounds[name]:
                    continue

                vals = bounds[name][coeff]
                if vals["mean_err95"] == 0.0:
                    continue
                ax.errorbar(
                    X[cnt] + val[i],
                    y=np.array(vals["mid"]),
                    yerr=np.array([vals["err95"]]).T,
                    color=colors(2 * i + 1),
                )
                ax.errorbar(
                    X[cnt] + val[i],
                    y=np.array(vals["mid"]),
                    yerr=np.array([vals["err68"]]).T,
                    color=colors(2 * i),
                    fmt=".",
                    label=label,
                )
                label = None
                # double soluton
                if f"{coeff}_2" in bounds[name]:
                    ax.errorbar(
                        X[cnt] + val[i],
                        y=np.array(bounds[name][f"{coeff}_2"]["mid"]),
                        yerr=np.array([bounds[name][f"{coeff}_2"]["err95"]]).T,
                        color=colors(2 * i + 1),
                    )
                    ax.errorbar(
                        X[cnt] + val[i],
                        y=np.array(bounds[name][f"{coeff}_2"]["mid"]),
                        yerr=np.array([bounds[name][f"{coeff}_2"]["err68"]]).T,
                        color=colors(2 * i),
                        fmt=".",
                    )

        py.plot(list(range(-1, 200)), np.zeros(201), "k--", alpha=0.7)

        py.yscale("symlog", linthresh=1e-1)
        py.ylim(-400, 400)
        py.yticks(
            [-100, -10, -1, -0.1, 0, 0.1, 1, 10, 100],
            [
                r"$-100$",
                r"$-10$",
                r"$-1$",
                r"$-0.1$",
                r"$0$",
                r"$0.1$",
                r"$1$",
                r"$10$",
                r"$100$",
            ],
        )
        py.ylabel(r"$c_i/\Lambda^2\ ({\rm TeV}^{-2})$", fontsize=18)

        py.xlim(-1, (self.npar) * 2 - 1)
        py.tick_params(which="major", direction="in", labelsize=13)
        py.xticks(X, self.coeff_labels, fontsize=10, rotation=45)

        py.legend(loc=0, frameon=False, prop={"size": 13})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Central.pdf")

    def plot_coeffs_bar(self, error):
        """
        Plot error bars at given confidence level

        Parameters
        ----------
            error: dict
               confidence level bounds per fit and coefficient
        """

        py.figure(figsize=(12, 6))
        df = pd.DataFrame.from_dict(error, orient="index", columns=self.coeff_labels).T
        df.plot(kind="bar", rot=0, width=0.6, figsize=(12, 5))

        # Hard cutoff
        py.plot(
            np.linspace(-1, 2 * self.npar + 1, 2),
            400 * np.ones(2),
            "k--",
            alpha=0.7,
            lw=2,
        )
        py.xticks(fontsize=10, rotation=45)
        py.tick_params(axis="y", direction="in", labelsize=15)
        py.yscale("log")
        py.ylabel(
            r"$95\%\ {\rm Confidence\ Level\ Bounds}\ (1/{\rm TeV}^2)$", fontsize=11
        )
        py.ylim(1e-3, 4e3)
        py.legend(loc=2, frameon=False, prop={"size": 11})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Bar.pdf")

    def plot_residuals_bar(self, residual):
        """
        Plot residuals at given confidence level

        Parameters
        ----------
            residual: dict
                residuals per fit and coefficient
        """

        py.figure(figsize=(12, 6))
        df = pd.DataFrame.from_dict(
            residual, orient="index", columns=self.coeff_labels
        ).T

        ax = df.plot(kind="bar", rot=0, width=0.6, figsize=(12, 5))
        ax.plot([-1, self.npar + 1], np.zeros(2), "k--", lw=2)
        ax.plot([-1, self.npar + 1], np.ones(2), "k--", lw=2, alpha=0.3)
        ax.plot([-1, self.npar + 1], -1.0 * np.ones(2), "k--", lw=2, alpha=0.3)

        py.xticks(fontsize=10, rotation=45)
        py.tick_params(axis="y", direction="in", labelsize=15)
        py.ylabel(r"${\rm Residuals\ (68\%)}$", fontsize=15)
        py.ylim(-3, 3)
        py.legend(loc=2, frameon=False, prop={"size": 11})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Residuals.pdf")

    def plot_posteriors(self, posteriors, labels, disjointed_lists=None):
        """ " Plot posteriors histograms

        Parameters
        ----------
            posteriors : dict
                posterior distibutions per fit and coefficent
            labels : list
                list of fit names
            disjointed_list: list, optional
                list of coefficients with double solutions
        """  # pylint:disable=import-outside-toplevel
        import warnings
        import matplotlib

        warnings.filterwarnings("ignore", category=matplotlib.mplDeprecation)

        colors = py.rcParams["axes.prop_cycle"].by_key()["color"]

        gs = int(np.sqrt(self.npar)) + 1
        nrows, ncols = gs, gs
        fig = py.figure(figsize=(nrows * 4, ncols * 3))
        for clr_cnt, posterior in enumerate(posteriors.values()):
            for cnt, l in enumerate(self.coeff_list):
                ax = py.subplot(ncols, nrows, cnt + 1)
                solution = posterior[l]
                if solution.all() == 0.0:
                    continue
                if disjointed_lists is None:
                    pass
                elif l in disjointed_lists[clr_cnt]:
                    solution, solution2 = utils.split_solution(posterior[l])
                    ax.hist(
                        solution2,
                        bins="fd",
                        density=True,
                        color=colors[clr_cnt],
                        edgecolor="black",
                        alpha=0.3,
                    )
                ax.hist(
                    solution,
                    bins="fd",
                    density=True,
                    color=colors[clr_cnt],
                    edgecolor="black",
                    alpha=0.3,
                    label=labels[clr_cnt],
                )
                # if clr_cnt == 0 :
                ax.text(
                    0.05,
                    0.85,
                    utils.latex_coeff(l),
                    transform=ax.transAxes,
                    fontsize=20,
                )

                ax.tick_params(which="both", direction="in", labelsize=22.5)
                ax.tick_params(labelleft=False)

        lines, labels = fig.axes[0].get_legend_handles_labels()
        for axes in fig.axes:
            if len(axes.get_legend_handles_labels()[0]) > len(lines):
                lines, labels = axes.get_legend_handles_labels()
        fig.legend(lines, labels, loc="lower right", prop={"size": 35})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Hist.pdf")

    def plot_single_posterior(self, coeff_name, posteriors, bounds, labels):
        """ " Plot posteriors histograms

        Parameters
        ----------
            coeff_name : str
                selected coefficient name
            posteriors : dict
                posterior distibutions per fit and coefficent
            bounds : dict
                Confidence level bounds per fit and coefficent
            labels: list
                list of fit names
        """
        # colors = py.rcParams["axes.prop_cycle"].by_key()["color"]
        colors = cm.get_cmap("tab20")

        fig = py.figure(figsize=(7, 5))
        gs = fig.add_gridspec(5, 1)
        ax = py.subplot(gs[:-1])
        ax_ratio = py.subplot(gs[-1])

        for clr_cnt, (name, posterior) in enumerate(posteriors.items()):
            solution = posterior[coeff_name]
            if solution.all() == 0.0:
                continue
            label = labels[clr_cnt]
            vals = bounds[label][coeff_name]
            # cl bounds
            if f"{coeff_name}_2" in bounds[label]:
                solution, solution2 = utils.split_solution(posterior[coeff_name])
                ax.hist(
                    solution2,
                    bins="fd",
                    density=True,
                    color=colors(2 * clr_cnt),
                    edgecolor="black",
                    alpha=0.3,
                )
                vals2 = bounds[label][f"{coeff_name}_2"]
                ax_ratio.errorbar(
                    x=np.array(vals2["mid"]),
                    y=clr_cnt,
                    xerr=np.array([vals2["err95"]]).T,
                    color=colors(2 * clr_cnt + 1),
                    elinewidth=3,
                )
                ax_ratio.errorbar(
                    x=np.array(vals2["mid"]),
                    y=clr_cnt,
                    xerr=np.array([vals2["err68"]]).T,
                    color=colors(2 * clr_cnt),
                    elinewidth=3,
                    fmt=".",
                )
            ax.hist(
                solution,
                bins="fd",
                density=True,
                color=colors(2 * clr_cnt),
                edgecolor="black",
                alpha=0.3,
                label=name,
            )
            ax_ratio.errorbar(
                x=np.array(vals["mid"]),
                y=clr_cnt,
                xerr=np.array([vals["err95"]]).T,
                color=colors(2 * clr_cnt + 1),
                elinewidth=3,
            )
            ax_ratio.errorbar(
                x=np.array(vals["mid"]),
                y=clr_cnt,
                xerr=np.array([vals["err68"]]).T,
                color=colors(2 * clr_cnt),
                elinewidth=3,
                fmt=".",
            )

        ax.text(
            0.03,
            0.88,
            utils.latex_coeff(coeff_name),
            transform=ax.transAxes,
            fontsize=35,
        )
        ax.set_ylabel(r"${ \rm Posterior\ Distibution\ }$", fontsize=20)

        ax.legend(labels, loc=1, prop={"size": 10})
        ax.tick_params(color="black", labelsize=18, width=1)
        ax.set_xticklabels([])
        ax.set_yticks([])
        ax.set_yticklabels([])

        ax_ratio.set_ylabel(r"${ \rm 95\%~CL }$", fontsize=15)
        ax_ratio.set_yticklabels([])
        ax_ratio.set_yticks([])
        ax_ratio.set_xlim(ax.get_xlim())
        ax_ratio.set_ylim(-1, len(bounds))
        ax_ratio.plot(
            np.zeros(len(bounds) + 2),
            list(range(-1, len(bounds) + 1)),
            "k--",
            linewidth=2,
            alpha=0.7,
        )
        ax_ratio.tick_params(color="black", labelsize=18, width=1)
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Hist_{coeff_name}.pdf")

    def write_cl_table(self, bounds):
        """
        Write table with CL bounds

        Parameters
        ----------
            bounds: dict
                confidence level bounds per fit and coefficient
        """
        pd.set_option("display.max_colwidth", 999)
        pd.set_option("precision", 2)
        for name in bounds:
            df = pd.DataFrame(bounds[name]).T
            caption = f"{name} Confidence Level bounds"
            rich.print(df.to_latex(column_format="cccccccc", caption=caption))
