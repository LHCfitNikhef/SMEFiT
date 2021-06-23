# -*- coding: utf-8 -*-
import numpy as np
import subprocess


class Runner:  # pylint:disable=import-error,import-outside-toplevel, anomalous-backslash-in-string
    """
    Runner class for smefit lite

    Parameters
    ----------
        data_path : str
            data path
        report_name : str
            report directory name
        fits : list
            list of instances ofsmefit_lite.fit_manager.FitManger
    """

    def __init__(self, data_path, report_name, fits):
        """
        Init the data path where the rusults are stored

        """
        import pathlib

        print(20 * "  ", " ____  __  __ _____ _____ _ _____ ")
        print(20 * "  ", "/ ___||  \/  | ____|  ___(_)_   _|")
        print(20 * "  ", "\___ \| |\/| |  _| | |_  | | | |  ")
        print(20 * "  ", " ___) | |  | | |___|  _| | | | |  ")
        print(20 * "  ", "|____/|_|  |_|_____|_|   |_| |_|  ")
        print()
        print(18 * "  ", "A Standard Model Effective Field Theory Fitter")

        self.data_path = pathlib.Path(data_path).absolute()
        self.report_name = report_name
        self.report_folder = f"{self.data_path.parents[0]}/reports/{self.report_name}"
        self.fits = fits
        self.fit_labels = [fit.label for fit in self.fits]

    def _build_report_folder(self):
        """Construct results folder if deos not exists"""
        subprocess.call(f"mkdir -p {self.data_path.parents[0]}/reports/", shell=True)

        # Clean output folder if exists
        subprocess.call(f"rm -rf {self.report_folder}", shell=True)
        subprocess.call(f"mkdir {self.report_folder}", shell=True)

    def run(self, free_dofs=None, plot_only=None):
        """
        Run the analysis

        Parameters
        ----------
            free_dofs : dict, optional
                dictionary with hidden and visible degrees of freedom
            plot_only : list, optional
                chose some specific plots to produce within:
                'cl_vals', 'cl_bars', 'residuals', 'post_hist', 'coeff_table', 'correlations'
                if equal to a name of a coefficient plot the single posterior histogram
                if None plot all
        """
        from matplotlib import use
        from matplotlib import rc

        from .analyze.correlation import plot as corr_plot
        from .analyze.coefficients import CoefficientsPlotter
        from .fixed_dofs import propagate_constraints

        # global mathplotlib settings
        use("PDF")
        rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        rc("text", usetex=True)

        free_dofs = free_dofs or {"hide": [], "show": []}
        if plot_only is None:
            plot_only = [
                "cl_vals",
                "cl_bars",
                "residuals",
                "post_hist",
                "coeff_table",
                "correlations",
            ]

        # load results and configutation
        config = {}
        posteriors = {}
        for fit in self.fits:
            config.update({fit.label: fit.load_configuration()})
            posteriors.update({fit.label: fit.load_posterior()})

        print(2 * "  ", f"Loading: {[fit.name for fit in self.fits]}")
        self._build_report_folder()
        coeff_ptl = CoefficientsPlotter(config, self.report_folder, hide_dofs=free_dofs["hide"])

        # Build the confidence levels
        # They are stored in a dict per fit name and coefficient
        cl_bounds = {}
        disjointed_lists = []
        for name in self.fit_labels:
            disjointed_lists.append(config[name]["double_solution"])
            propagate_constraints(config[name], posteriors[name])
            cl_bounds[name] = coeff_ptl.compute_confidence_level(
                posteriors[name], 
                config[name]["double_solution"]
            )

        if "cl_vals" in plot_only:
            print(2 * "  ", "Plotting: Central values and Confidence Level bounds")
            coeff_ptl.plot_coeffs(cl_bounds, disjointed_lists)

        if "cl_bars" in plot_only:
            print(2 * "  ", "Plotting: Confidence Level error bars")
            temp = cl_bounds.copy()
            # Uncomment if you want to show the total error bar for double solution,
            # otherwhise show 95% CL for null solutions.
            # add second error if exists
            for name in self.fit_labels:
                for op in list(config[name]["double_solution"]):
                    temp[name][op]["cl95"] += temp[name][f"{op}_2"]["cl95"]

            coeff_ptl.plot_coeffs_bar(
                {
                    name: [
                        np.sum(temp[name][op]["cl95"]) for op in coeff_ptl.coeff_list
                    ]
                    for name in temp
                }
            )
        if "residuals" in plot_only:
            print(2 * "  ", "Plotting: Residuals")
            coeff_ptl.plot_residuals_bar(
                {
                    name: [
                        cl_bounds[name][op]["mid"] / cl_bounds[name][op]["error68"]
                        for op in coeff_ptl.coeff_list
                    ]
                    for name in cl_bounds
                }
            )

        if "post_hist" in plot_only:
            print(2 * "  ", "Plotting: Posterior histograms")
            coeff_ptl.plot_posteriors(
                posteriors,
                labels=self.fit_labels,
                disjointed_lists=[config[name]["double_solution"] for name in self.fit_labels],
            )

        if plot_only in coeff_ptl.coeff_list:
            print(2 * "  ", f"Plotting: {plot_only} Posterior histograms")
            coeff_ptl.plot_single_posterior(
                plot_only,
                posteriors,
                cl_bounds,
                labels=self.fit_labels
            )

        if "coeff_table" in plot_only:
            print(2 * "  ", "Writing: Confidence level table")
            coeff_ptl.write_cl_table(
                cl_bounds,
            )

        if "correlations" in plot_only:
            print(2 * "  ", "Plotting: Correlations")
            for fit in self.fits:
                corr_plot(
                    config[fit.label],
                    posteriors[fit.label],
                    f"{self.report_folder}/Coeffs_Corr_{fit.name}.pdf",
                    dofs=free_dofs,
                )
