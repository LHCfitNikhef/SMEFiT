# # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as py
import rich

from . import coefficients_utils as utils


class CoefficientsPlotter:
    """
    Plots central values + 95% CL errors, 95% CL bounds and residuals.

    Takes into account parameter constraints and displays
    all non-zero parameters.

    Parameters
    ----------
       config : dict
            configuration dictionary
    """

    def __init__(self, config, hide_dofs=None):

        hide_dofs = hide_dofs or []
        self.report_folder = (
            f"{config['data_path'].parents[0]}/reports/{config['report_name']}"
        )

        coeff_config = utils.coeff_by_group().copy()
        temp = config.copy()
        temp.pop("data_path")
        temp.pop("report_name")
        self.coeff_list = []
        for group in coeff_config.values():
            for c in group:
                if c in hide_dofs:
                    continue
                if np.any(
                    [
                        config[k]["coefficients"][c]["fixed"] is True
                        for k in temp.keys()
                        if c in config[k]["coefficients"]
                    ]
                ):
                    continue
                if c not in self.coeff_list and np.any(
                    [c in config[k]["coefficients"] for k in temp.keys()]
                ):
                    self.coeff_list.append(c)

        self.npar = len(self.coeff_list)

    def compute_confidence_level(self, posterior, disjointed_list = None):
        """
        Compute 95 % and 68 % confidence levels and store the result in a dictionary

        Parameters
        ----------
            posterior : dict
                posterior distibutions per coefficent
            disjointed_list: list, optional
                list of coefficients with double solutions

        Returns
        -------
            bounds: dict
                confidence level bounds per coefficient
                Note: double solutions are appended under "2"
        """

        disjointed_list = disjointed_list or []
        bounds = {}
        for l in self.coeff_list:
            posterior[l] = np.array(posterior[l])
            cl_vals = {}
            # double soultion
            if l in disjointed_list:
                cl_vals1, cl_vals2 = utils.get_double_cls(posterior[l])
                bounds.update({l: cl_vals1})
                bounds.update({f"{l}_2": cl_vals2})
            # single solution
            else:
                cl_vals = utils.get_conficence_values(posterior[l])
                bounds.update({l: cl_vals})

        return bounds

    def plot_coeffs(self, bounds, disjointed_lists):
        """
        Plot central value + 95% CL errors

        Parameters
        ----------
            bounds: dict
                confidence level bounds per fit and coefficient
                Note: double solutions are appended under "2"
        """

        nrows, ncols = 1, 1
        py.figure(figsize=(nrows * 10, ncols * 5))
        ax = py.subplot(111)

        # X-axis
        X = 2 * np.array(range(self.npar))
        # Spacing between fit results
        val = np.linspace(-0.1 * len(bounds), 0.1 * len(bounds), len(bounds))
        colors = py.rcParams["axes.prop_cycle"].by_key()["color"]

        # loop over fits
        for i, name in enumerate(bounds):
            for cnt, coeff in enumerate(self.coeff_list):
                if coeff not in bounds[name].keys():
                    continue
                vals = bounds[name][coeff]
                if cnt == 0:
                    label = name
                eb = ax.errorbar(
                    X[cnt] + val[i],
                    y=np.array(vals["mid"]),
                    yerr=np.array(vals["error95"]),
                    color=colors[i],
                    fmt=".",
                    elinewidth=1,
                    label=label,
                )
                eb[-1][0].set_linestyle(':')
                ax.errorbar(
                    X[cnt] + val[i],
                    y=np.array(vals["mid"]),
                    yerr=np.array(vals["error68"]),
                    color=colors[i],
                )
                label = None
                # double soluton
                if coeff in disjointed_lists[i]:
                    ax.errorbar(
                        X[cnt] + val[i],
                        y=np.array( bounds[name][f"{coeff}_2"]["mid"]),
                        yerr=np.array(bounds[name][f"{coeff}_2"]["error95"]),
                        color=colors[i],
                        fmt=".",
                        elinewidth=1,
                    )

        py.plot(list(range(-1, 200)), np.zeros(201), "k--", alpha=0.7)

        py.yscale("symlog", linthresh=1e-1)
        py.ylim(-200, 200)
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
        py.xticks(X, self.coeff_list, rotation=90)

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

        py.figure(figsize=(7, 5))
        df = pd.DataFrame.from_dict(error, orient="index", columns=self.coeff_list).T
        df.plot(kind="bar", rot=0, width=0.7, figsize=(10, 5))

        # Hard cutoff
        py.plot(
            np.linspace(-1, 2 * self.npar + 1, 2),
            50 * np.ones(2),
            "k--",
            alpha=0.7,
            lw=2,
        )
        py.xticks(rotation=90)
        py.tick_params(axis="y", direction="in", labelsize=15)
        py.yscale("log")
        py.ylabel(
            r"$95\%\ {\rm Confidence\ Level\ Bounds}\ (1/{\rm TeV}^2)$", fontsize=11
        )
        py.ylim(1e-3, 1e3)
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

        py.figure(figsize=(7, 5))
        df = pd.DataFrame.from_dict(residual, orient="index", columns=self.coeff_list).T

        ax = df.plot(kind="bar", rot=0, width=0.7, figsize=(10, 5))
        ax.plot([-1, self.npar + 1], np.zeros(2), "k--", lw=2)
        ax.plot([-1, self.npar + 1], np.ones(2), "k--", lw=2, alpha=0.3)
        ax.plot([-1, self.npar + 1], -1.0 * np.ones(2), "k--", lw=2, alpha=0.3)

        py.xticks(rotation=90)
        py.tick_params(axis="y", direction="in", labelsize=15)
        py.ylabel(r"${\rm Residuals\ (68\%)}$", fontsize=15)
        py.ylim(-3, 3)
        py.legend(loc=2, frameon=False, prop={"size": 11})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Residuals.pdf")

    def plot_posteriors(self, posteriors, disjointed_lists = None):
        """" Plot posteriors histograms

        Parameters
        ----------
            posterior : dict
                posterior distibutions per fit and coefficent
            disjointed_list: list, optional
                list of coefficients with double solutions
        """  # pylint:disable=import-error,import-outside-toplevel
        import warnings
        import matplotlib
        warnings.filterwarnings("ignore",category=matplotlib.mplDeprecation)

        colors = py.rcParams["axes.prop_cycle"].by_key()["color"]

        gs = int(np.sqrt(self.npar)) + 1
        nrows, ncols = gs, gs
        fig = py.figure(figsize=(nrows * 4, ncols * 3))

        total_cnt = []
        for clr_cnt, (name, posterior) in enumerate(posteriors.items()):
            for cnt, l in enumerate(self.coeff_list):
                cnt += 1
                ax = py.subplot(ncols, nrows, cnt)
                solution = posterior[l]
                if l in disjointed_lists[clr_cnt]:
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
                    label=r"${\rm %s}$"
                    % name.replace(
                        "_", "\ "  # pylint:disable=anomalous-backslash-in-string
                    ),
                )
                if clr_cnt == 0:
                    ax.text(
                        0.05,
                        0.85,
                        r"${\rm {\bf " + l + "}}$",
                        transform=ax.transAxes,
                        fontsize=20,
                    )

                ax.tick_params(which="both", direction="in", labelsize=22.5)
                ax.tick_params(labelleft=False)

                if cnt not in total_cnt:
                    total_cnt.append(cnt)

        lines, labels = fig.axes[-2].get_legend_handles_labels()
        fig.legend(lines, labels, loc="lower right", prop={"size": 35})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Hist.pdf")

    def write_cl_table(self, bounds):
        """
        Write table with CL bounds

        Parameters
        ----------
            bounds: dict
                confidence level bounds per fit and coefficient
        """
        pd.set_option('display.max_colwidth', None)
        pd.set_option('precision', 2)
        for name in bounds:
            df = pd.DataFrame(bounds[name]).T
            caption = f"{name} Confidence Level bounds"
            rich.print(df.to_latex(column_format='cccccccc', caption=caption))
