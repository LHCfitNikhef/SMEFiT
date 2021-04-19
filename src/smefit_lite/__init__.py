# -*- coding: utf-8 -*-
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
            list of fits id
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

    def _load_configurations(self):
        """Load configuration yaml card for each fit

        Returns
        -------
            config: dict
                configuration card
        """
        import yaml

        config = {}
        config.update({"data_path": self.data_path, "report_name": self.report_name})
        for fit in self.fits:
            with open(f"{self.data_path}/{fit}/{fit}.yaml") as f:
                temp = yaml.safe_load(f)
            config.update({fit: temp})
        return config

    def _load_posteriors(self):
        """Load posterior distribution for each fit

        Returns
        -------
            posterior : dict
                posterior distibutions per fit and coefficent
        """
        import json

        posterior = {}
        for fit in self.fits:
            with open(f"{self.data_path}/{fit}/posterior.json") as f:
                temp = json.load(f)
            posterior.update({fit: temp})
        return posterior

    def _build_report_folder(self):
        """ Construct results folder if deos not exists

        Parameters
        ----------
            report_name : str
                report name
        """
        subprocess.call(f"mkdir -p {self.data_path.parents[0]}/reports/", shell=True)

        # Clean output folder if exists
        subprocess.call(f"rm -rf {self.report_folder}", shell=True)
        subprocess.call(f"mkdir {self.report_folder}", shell=True)

    def run(self, free_dofs=None):
        """
        Run the analysis

        Parameters
        ----------
            free_dofs : dict, optional
                dictionary with hidden and visible degrees of freedom
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

        free_dofs = free_dofs or {}
        config = self._load_configurations()
        posteriors = self._load_posteriors()

        print(2 * "  ", f"Loading: {self.fits}")
        self._build_report_folder()
        coeff_ptl = CoefficientsPlotter(config, hide_dofs=free_dofs["hide"])

        # Build the confidence levels
        cl_bounds = {}
        disjointed_lists = []
        for k in self.fits:
            name = r"${\rm %s}$" % k.replace(
                "_", "\ "
            )
            disjointed_lists.append( config[k]["double_solution"] )
            propagate_constraints(config[k], posteriors[k])
            cl_bounds[name] = coeff_ptl.compute_confidence_level(
                posteriors[k], config[k]["double_solution"]
            )

        print(2 * "  ", "Plotting: Central values and Confidence Level bounds")
        coeff_ptl.plot_coeffs(cl_bounds, disjointed_lists)

        print(2 * "  ", "Plotting: Confidence Level error bars")

        # Uncomment if you want to show the total error bar for double solution,
        # otherwhise show 95% CL for null solutions.
        # add second error if exists
        # for k in self.fits:
        #     name = r"${\rm %s}$" % k.replace(
        #         "_", "\ "
        #     )
        #     for op in list(config[k]["double_solution"]):
        #         cl_bounds[name][op]["error95"] += cl_bounds[name][op]["2"]["error95"]

        coeff_ptl.plot_coeffs_bar(
            {
                name: [cl_bounds[name][op]["error95"] for op in coeff_ptl.coeff_list ]
                for name in cl_bounds
            }
        )

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

        print(2 * "  ", "Plotting: Posterior histograms")
        coeff_ptl.plot_posteriors(
            posteriors,
            disjointed_lists=[config[k]["double_solution"] for k in self.fits],
        )

        print(2 * "  ", "Writing: Confidence level table")
        coeff_ptl.write_cl_table(
            cl_bounds,
        )

        print(2 * "  ", "Plotting: Correlations")
        for k in self.fits:
            corr_plot(
                config[k],
                posteriors[k],
                f"{self.report_folder}/Coeffs_Corr_{k}.pdf",
                dofs=free_dofs,
            )
