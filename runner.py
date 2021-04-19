import pathlib
import smefit_lite

if __name__ == "__main__":


    path = f"{pathlib.Path(__file__).parents[0].absolute()}/SMEFiT20"
    fit = ["NS_GLOBAL_NLO_NHO"]
    report_name = "test"

    smefit = smefit_lite.Runner(path, report_name, fit)
    smefit.run()
