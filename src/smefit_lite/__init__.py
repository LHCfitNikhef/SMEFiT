# -*- coding: utf-8 -*-
import subprocess


class RUNNER:  # pylint:disable=import-error,import-outside-toplevel
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

        """  # pylint:disable=anomalous-backslash-in-string
        import pathlib

        print(20 * "  ", " ____  __  __ _____ _____ _ _____ ")
        print(20 * "  ", "/ ___||  \/  | ____|  ___(_)_   _|")
        print(20 * "  ", "\___ \| |\/| |  _| | |_  | | | |  ")
        print(20 * "  ", " ___) | |  | | |___|  _| | | | |  ")
        print(20 * "  ", "|____/|_|  |_|_____|_|   |_| |_|  ")
        print()
        print(18 * "  ", "A Standar Model Effective Field Theory Fitter")

        self.data_path = pathlib.Path(data_path).absolute()
        self.report_name = report_name
        self.report_folder = f"{self.data_path.parents[1]}/reports/{self.report_name}"
        self.fits = fits

    def _load_configurations(self):
        """Load configuration yaml card for each fit

        Returns
        -------
            config: dict
                configuration dict
        """
        import yaml

        config = {}
        config["data_path"] = self.data_path
        config["report_name"] = self.report_folder
        for fit in self.fits:
            config[fit] = {}
            with open(f"{self.data_path}/{fit}.yaml") as f:
                config = yaml.safe_load(f)
        return config

    def _load_posteriors(self):
        """Load posterior distribution for each fit

        Returns
        -------
            posterior : dict
                dictionary containging the posterior distibution
        """
        import json

        posterior = {}
        for fit in self.fits:
            posterior[fit] = {}
            with open(f"{self.data_path}/{fit}/posterior.json") as f:
                posterior = json.load(f)
        return posterior

    def _build_report_folder(self):
        """ Construct results folder if deos not exists

        Parameters
        ----------
            report_name : str
                report name
        """
        subprocess.call(f"mkdir -p {self.data_path.parents[1]}/reports/", shell=True)

        # Clean output folder if exists
        report_folder = f"{self.data_path.parents[1]}/reports/{self.report_name}"
        try:
            subprocess.call(f"rm -rf {report_folder}", shell=True)
        except FileNotFoundError:
            subprocess.call(f"mkdir {report_folder}", shell=True)

    def _move_to_meta(self):
        """Move pdf files to meta folder"""

        subprocess.call(f"mkdir -p {self.report_folder}/meta", shell=True)
        subprocess.call(
            f"mv {self.report_folder}/*.pdf {self.report_folder}/meta/", shell=True,
        )

    def _write_report(self):
        """Combine all plots into a single report"""
        from PyPDF2 import PdfFileReader, PdfFileWriter

        # TODO: Combine PDF files together into pdf report
        # TODO: add a summary of the settings

        report_pdf = f"{self.report_folder}/report_{self.report_name}"
        flags = "-q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite "
        subprocess.call(
            f"gs {flags} -sOutputFile={report_pdf}_raw.pdf `ls -rt {self.report_folder}/*.pdf`",
            shell=True,
        )
        subprocess.call(
            f"mv {self.report_folder}/*.* {self.report_folder}/meta/.", shell=True
        )
        subprocess.call(
            f"mv {self.report_folder}/meta/report_*.pdf  {self.report_folder}/.",
            shell=True,
        )

        # Rotate PDF pages if necessary and create final report
        pdf_in = open(f"{report_pdf}_raw.pdf", "rb")
        pdf_reader = PdfFileReader(pdf_in)
        pdf_writer = PdfFileWriter()
        for pagenum in range(pdf_reader.numPages):
            pdfpage = pdf_reader.getPage(pagenum)
            orientation = pdfpage.get("/Rotate")
            if orientation == 90:
                pdfpage.rotateCounterClockwise(90)
            pdf_writer.addPage(pdfpage)
        pdf_out = open(f"{report_pdf}.pdf", "wb")
        pdf_writer.write(pdf_out)
        pdf_out.close()
        pdf_in.close()

        # Remove old (raw) PDF file
        subprocess.call(
            f"rm {self.report_folder}/report_{self.report_name}_raw.pdf", shell=True
        )

    def run(self):
        """Run the analysis"""
        from matplotlib import use
        from matplotlib import rc

        from .analyze.correlation import plot as corr_plot
        from .analyze.coefficients import CoefficientsPlotter

        # global mathplotlib settings
        use("PDF")
        rc("font", **{"family": "sans-serif", "sans-serif": ["Helvetica"]})
        rc("text", usetex=True)

        config = self._load_configurations()
        posteriors = self._load_posteriors()
        self._build_report_folder()

        coeff_ptl = CoefficientsPlotter(config)
        residuals = {}
        cl_bounds = {}
        error_bounds = {}

        for k in self.fits:
            disjointed_list = list(config[k]["double_solution"])
            name = r"${\rm %s}$" % k.replace(
                "_", "\_" # pylint:disable=anomalous-backslash-in-string
            )
            # TODO: add here a function to set the constrains for each fit
            (
                cl_bounds[name],
                residuals[name],
                error_bounds[name],
            ) = coeff_ptl.compute_confidence_level(posteriors, disjointed_list)

        coeff_ptl.plot_coeffs(cl_bounds)
        coeff_ptl.plot_coeffs_bar(error_bounds)
        coeff_ptl.plot_residuals_bar(residuals)

        # correlation plots
        for k in self.fits:
            # report.plot_coefficients()
            corr_plot(k, config[k], posteriors[k])

        self._move_to_meta()
        self._write_report()
