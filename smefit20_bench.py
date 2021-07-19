import pathlib
from smefit_lite import Runner
from smefit_lite.fit_manager import FitManager

path = pathlib.Path().absolute()
report_path = f"{path}/reports"


def Sfitter_bench():
    """SMEFiT 2.0 vs SFitter NLO, Quadratic comparison"""
    fit_smefit = FitManager(
        f"{path}/SMEFiT20",
        "NS_GLOBAL_NLO_HO",
        label=r"${\rm SMEFiT\ Top+H+VV,\ NLO\ QCD\ \mathcal{O}(\Lambda^4)}$",
    )
    fit_sfitter = FitManager(
        f"{path}/external",
        "SFitter_TOP_NLO_HO",
        label=r"${\rm SFitter\ Top-only,\ NLO\ QCD\ \mathcal{O}(\Lambda^4)}$",
        has_posterior=False,
    )
    report_name = "sfitter_bench"

    smefit = Runner(f"{report_path}/{report_name}", [fit_smefit, fit_sfitter])
    smefit.run(
        free_dofs={"show": ["cpWB", "cpD"], "hide": ["cB", "cW"]},
        plot_only=[
            "cl_vals",
            "cl_bars",
            "residuals",
            "coeff_table",
        ],
    )

class FitMarker_runner():
    """"FitMaker settings"""

    def __init__(self):
        self.plot_only = [
                "cl_vals",
                "cl_bars",
                "residuals",
                "coeff_table",
            ]
        self.free_dofs = {"show": ["cpWB", "cpD"], "hide": ["cB", "cW", "ctB", "cpQ", "cpqi"]}

    def marginalized_bench(self):
        """SMEFiT 2.0 vs FitMarker LO, Linear comparison"""
        fit_smefit = FitManager(
            f"{path}/SMEFiT20",
            "NS_GLOBAL_LO_NHO",
            label=r"${\rm SMEFiT\ Top+H+VV,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
        )
        fit_fitmarker = FitManager(
            f"{path}/external",
            "FitMaker_GLOBAL_LO_NHO",
            label=r"${\rm FitMaker\ Top+H+VV,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
            has_posterior=False,
        )
        report_name = "fitmaker_bench"

        smefit = Runner(f"{report_path}/{report_name}", [fit_smefit, fit_fitmarker])
        smefit.run( free_dofs=self.free_dofs, plot_only= self.plot_only)


    def individual_bench(self):
        """SMEFiT 2.0 vs FitMarker Individual LO, Linear comparison"""
        fit_smefit = FitManager(
            f"{path}/SMEFiT20",
            "SNS_GLOBAL_LO_NHO",
            label=r"${\rm SMEFiT\ Top+H+VV\ Individual,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
            is_individual=True
        )
        fit_fitmarker = FitManager(
            f"{path}/external",
            "FitMaker_INDIV_LO_NHO",
            label=r"${\rm FitMaker\ Top+H+VV\ Individual,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
            has_posterior=False,
        )
        report_name = "fitmaker_individual_bench"

        smefit = Runner(f"{report_path}/{report_name}", [fit_smefit, fit_fitmarker])
        smefit.run( free_dofs=self.free_dofs, plot_only= self.plot_only)

if __name__ == "__main__":

    Sfitter_bench()
    runner = FitMarker_runner()
    runner.marginalized_bench()
    runner.individual_bench()