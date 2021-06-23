import pathlib
import smefit_lite

path = f"{pathlib.Path(__file__).parents[0].absolute()}/SMEFiT20"


def report_test():

    fit = ["NS_GLOBAL_NLO_NHO", "NS_GLOBAL_NLO_HO"]
    #labels=[r"${\rm LO~QCD,~Quadratic~EFT}$",r"${\rm NLO~QCD,~Quadratic~EFT}$"]
    report_name = "test"

    smefit = smefit_lite.Runner(path, report_name, fit,)
    smefit.run( free_dofs = {"show": ["cpWB", "cpD"], "hide": ["cB", "cW"]} )

def plot_hist_only(coeff_name):

    fit = ["NS_GLOBAL_NLO_NHO", "NS_GLOBAL_NLO_HO"]
    labels=[r"${\rm LO~QCD,~Quadratic~EFT}$",r"${\rm NLO~QCD,~Quadratic~EFT}$"]

    report_name = f"test_{coeff_name}"

    smefit = smefit_lite.Runner(path, report_name, fit, fit_labels=labels)
    smefit.run( plot_only = coeff_name )

if __name__ == "__main__":

    report_test()
    plot_hist_only('ctG')