import pathlib
from smefit_lite import Runner
from smefit_lite.fit_manager import FitManager


data_path = f"{pathlib.Path().absolute()}/SMEFiT20"
report_path = f"{pathlib.Path().absolute()}/reports"


def report_test():

    fit_linear = FitManager(
        data_path, "NS_GLOBAL_NLO_NHO", label=r"${\rm NLO~QCD,~Linear~EFT}$"
    )
    fit_quad = FitManager(
        data_path, "NS_GLOBAL_NLO_HO",
    )
    report_name = "test"

    smefit = Runner(f"{report_path}/{report_name}",  [fit_linear, fit_quad])
    smefit.run( free_dofs = {"show": ["cpWB", "cpD"], "hide": ["cB", "cW"]} )




def plot_hist_only(coeff_name):

    fit_linear = FitManager(
        data_path, "NS_GLOBAL_NLO_NHO", label=r"${\rm NLO~QCD,~Linear~EFT}$"
    )
    fit_quad = FitManager(
        data_path, "NS_GLOBAL_NLO_HO", label=r"${\rm NLO~QCD,~Quadratic~EFT}$"
    )

    report_name = f"/test_{coeff_name}"

    smefit = Runner(f"{report_path}/{report_name}",  [fit_linear, fit_quad])
    smefit.run( plot_only = coeff_name )

if __name__ == "__main__":

    report_test()
    plot_hist_only('ctG')
