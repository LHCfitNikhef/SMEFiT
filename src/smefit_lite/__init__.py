# -*- coding: utf-8 -*-
class RUNNER:  # pylint:disable=import-error,import-outside-toplevel
    """ Class containing all the possible smefit functions
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
        self.fits = fits

    def _load_results(self):
        """Read yaml card and posterior file

        Parameters
        ----------
            filename : str
                fit card name
        Returns
        -------
            config: dict
                configuration dict
            posterior : dict
                dictionary containging the posterior distibution
        """
        import json
        import yaml

        config = {}
        posterior = {}

        for fit in self.fits:
            config[fit] = {}
            posterior[fit] = {}
            # Load configuration
            with open(f"{self.data_path}/{fit}.yaml") as f:
                config = yaml.safe_load(f)

            # Load results
            with open(f"{self.data_path}/{fit}/posterior.json") as f:
                posterior = json.load(f)

        return config, posterior

    def _build_report(self):
        """ Construct results folder if deos not exists

        Parameters
        ----------
            report_name : str
                report name
        """
        import subprocess

        subprocess.call(f"mkdir -p {self.data_path.parents[1]}/reports/", shell=True)

        # Clean output folder if exists
        report_folder = f"{self.data_path.parents[1]}/reports/{self.report_name}"
        try:
            subprocess.call(f"rm -rf {report_folder}", shell=True)
        except FileNotFoundError:
            subprocess.call(f"mkdir {report_folder}", shell=True)

    def write_pdf(self):
        """Combine all plots into a single report"""
        import subprocess
        from PyPDF2 import PdfFileReader, PdfFileWriter

        # Combine PDF files together into raw pdf report
        report_folder = f"{self.data_path.parents[1]}/reports/{self.report_name}"
        report_pdf = f"{report_folder}/report_{self.report_name}"
        flags = "-q -dNOPAUSE -dBATCH -sDEVICE=pdfwrite "
        subprocess.call(
            f"gs {flags} -sOutputFile={report_pdf}_raw.pdf `ls -rt {report_folder}/*.pdf`",
            shell=True,
        )
        subprocess.call(f"mv {report_folder}/*.* {report_folder}/meta/.", shell=True)
        subprocess.call(
            f"mv {report_folder}/meta/report_*.pdf  {report_folder}/.", shell=True
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
            f"rm {report_folder}/report_{self.report_name}_raw.pdf", shell=True
        )

    def run(self):
        """Run the analysis"""
        from .analyze import Analyzer

        self._build_report()
        report = Analyzer(self.data_path, self.report_name, self._load_results())

        # Things to include in repor
        report.summary()
        report.coefficients()
        report.correlations()
