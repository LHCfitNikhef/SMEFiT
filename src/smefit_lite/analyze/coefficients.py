# # -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

import matplotlib.pyplot as py

from . import coefficients_utils as utils


class CoefficientsPlotter:
    """
    Plots central values + 95% CL errors, 95% CL bounds and residuals.

    Takes into account parameter constraints and displays
    all non-zero parameters.

    Note: coefficients that are known to have disjoint
    probability distributions (i.e. multiple solutions)
    are manually separated by including the coefficient name
    in disjointed_list for global.

    Note: Probability distributions for single parameter
    fits are included for all coefficients EXCEPT those
    constrained as a linear combination of two or more parameters
    (since they have different numbers of posterior samples)

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
                        config[k]["coefficients"][c]["fixed"] is not False
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

    def compute_confidence_level(self, fit, disjointed_list=None):
        """Compute 95 % and 68 % confidence levels and store everthing in a dictionary"""

        disjointed_list = disjointed_list or []
        bounds = {}
        for l in self.coeff_list:
            fit[l] = np.array(fit[l])
            cl_vals = {}
            # double soultion
            if l in disjointed_list:
                cl_vals = utils.set_double_cl(fit[l], l)
            # single solution
            else:
                cl_vals = utils.get_conficence_values(fit[l])
            bounds.update({l: cl_vals})

        return bounds

    def plot_coeffs(self, bounds):
        """Plot central value + 95% CL errors"""

        nrows, ncols = 1, 1
        py.figure(figsize=(nrows * 10, ncols * 5))
        ax = py.subplot(111)

        # X-axis
        X = 2 * np.array(range(self.npar))
        # Spacing between fit results
        val = np.linspace(-0.1 * self.npar, 0.1 * self.npar, self.npar)
        colors = py.rcParams["axes.prop_cycle"].by_key()["color"]

        i = 0
        # loop over fits
        for name in bounds:
            cnt = 0
            for vals in bounds[name].values():
                if cnt == 0:
                    label = name
                ax.errorbar(
                    X[cnt] + val[i],
                    y=np.array(vals["mid"]),
                    yerr=np.array(vals["error95"]),
                    color=colors[i],
                    fmt=".",
                    elinewidth=2,
                    label=label,
                )
                label = None
                # double soluton
                if "2" in vals.keys():
                    ax.errorbar(
                        X[cnt] + val[i] + 0.05,
                        y=np.array(vals["2"]["mid"]),
                        yerr=np.array(vals["2"]["error95"]),
                        color=colors[i],
                        fmt=".",
                        elinewidth=2,
                    )
                cnt += 1
            i += 1

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
        """ Plot 95% CLs for coefficients (bar plot)"""

        py.figure(figsize=(7, 5))
        df = pd.DataFrame.from_dict(error, orient="index", columns=self.coeff_list)
        new_df = df.T
        new_df.plot(kind="bar", rot=0, width=0.7, figsize=(10, 5))

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
        """ Plot residuals at 68% CL (bar plot) """
        py.figure(figsize=(7, 5))

        df = pd.DataFrame.from_dict(residual, orient="index", columns=self.coeff_list)
        new_df = df.T

        ax = new_df.plot(kind="bar", rot=0, width=0.7, figsize=(10, 5))
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

    def plot_posteriors(self, fits):
        """" Plot posteriors (histograms)"""

        nrows, ncols = 1, 1
        colors = py.rcParams["axes.prop_cycle"].by_key()["color"]

        gs = int(np.sqrt(self.npar)) + 1
        cnt = 1
        nrows, ncols = gs, gs
        fig = py.figure(figsize=(nrows * 4, ncols * 3))

        heights = {}
        total_cnt = []
        clr_cnt = 0
        for name, fit in fits.items():
            heights[name] = {}
            for l in self.coeff_list:
                ax = py.subplot(ncols, nrows, cnt)

                # if l in disjointed_list:
                #     min_val = min(fit[l])
                #     max_val = max(fit[l])
                #     mid = (max_val+min_val)/2.

                #     if l in ['Obp', 'Opd']:
                #         solution1 = fit[fit[l]>mid]
                #         solution2 = fit[fit[l]<mid]
                #     else:
                #         print(fit[l])
                #         solution1 = fit[fit[l]<mid]
                #         solution2 = fit[fit[l]>mid]
                #     heights[name][cnt]=ax.hist(solution1,bins='fd',density=True,color=colors[clr_cnt],edgecolor='black',alpha=0.3)
                #     ax.hist(solution2,bins='fd',density=True,color=colors[clr_cnt],edgecolor='black',alpha=0.3)
                # else:
                heights[name][cnt] = ax.hist(
                    fit[l],
                    bins="fd",
                    density=True,
                    color=colors[clr_cnt],
                    edgecolor="black",
                    alpha=0.3,
                    label=r"${\rm %s}$" % name.replace("_", "\ "),
                )  # pylint:disable=anomalous-backslash-in-string
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

                cnt += 1
            clr_cnt += 1

        lines, labels = fig.axes[-2].get_legend_handles_labels()
        fig.legend(lines, labels, loc="lower right", prop={"size": 35})
        py.tight_layout()
        py.savefig(f"{self.report_folder}/Coeffs_Hist.pdf")
