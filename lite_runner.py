import pathlib
from smefit_lite import Runner
from smefit_lite.fit_manager import FitManager

path = f"{pathlib.Path(__file__).parents[0].absolute()}/SMEFiT20"


def report_test():

    fit_linear = FitManager(
        path, "NS_GLOBAL_NLO_NHO", label=r"${\rm NLO~QCD,~Linear~EFT}$"
    )
    fit_quad = FitManager(
        path, "NS_GLOBAL_NLO_HO",
    )
    report_name = "test"

    smefit = Runner(path, report_name,  [fit_linear, fit_quad])
    smefit.run( free_dofs = {"show": ["cpWB", "cpD"], "hide": ["cB", "cW"]} )




def plot_hist_only(coeff_name):

    fit_linear = FitManager(
        path, "NS_GLOBAL_NLO_NHO", label=r"${\rm NLO~QCD,~Linear~EFT}$"
    )
    fit_quad = FitManager(
        path, "NS_GLOBAL_NLO_HO", label=r"${\rm NLO~QCD,~Quadratic~EFT}$"
    )

    report_name = f"test_{coeff_name}"

    smefit = Runner(path, report_name,  [fit_linear, fit_quad])
    smefit.run( plot_only = coeff_name )

if __name__ == "__main__":

    report_test()
    plot_hist_only('ctG')
