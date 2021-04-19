import pathlib
import smefit_lite

if __name__ == "__main__":


    path = f"{pathlib.Path(__file__).parents[0].absolute()}/SMEFiT20"
    fit = ["NS_GLOBAL_NLO_NHO", "NS_GLOBAL_NLO_HO"]
    report_name = "test"

    smefit = smefit_lite.Runner(path, report_name, fit)
    smefit.run( free_dofs = {"show": ["cpWB", "cpD"], "hide": ["cB", "cW"]} )
