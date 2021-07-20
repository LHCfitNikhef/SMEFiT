import pathlib
from smefit_lite import Runner
from smefit_lite.fit_manager import FitManager

path = pathlib.Path().absolute()
report_path = f"{path}/reports"


def Individual_bench():
    """ "
        VV+VBS settings

    IMOPRTANT note:
        the coeffientes cpe, cpl1, c3pl1, cpui, cpdi, cpqi, cpq3
        here follow the SMEFiT2.0 notation to have a consistent code,
        but they are not exactly the very same.
        See the different flavour assumptions of SMEFiT2.0 paper and VBS_SMEFiT paper.
        Inside the VBS_SFEFiT paper they are called: cpe, cpl, c3pl, cpu, cpd, cpq, cpq3.
        In paricular cpe, cpl1 and c3pl1 here are the average of the flavourful coefficients.
        cpui, cpdi, cpqi, cpq3 are the same as in the SMEFiT2.0 since top/b quark do not contribute
        any role in VBS, VV process

    """
    plot_only = [
        "cl_vals",
        "cl_bars",
        "coeff_table",
    ]
    free_dofs = {
        "hide": [
            "cB",
            "cW",
            "cQQ1",
            "cQQ8",
            "cQt1",
            "cQt8",
            "cQb1",
            "cQb8",
            "ctt1",
            "ctb1",
            "ctb8",
            "cQtQb1",
            "cQtQb8",
            "c81qq",
            "c11qq",
            "c83qq",
            "c13qq",
            "c8qt",
            "c1qt",
            "c8ut",
            "c1ut",
            "c8qu",
            "c1qu",
            "c8dt",
            "c1dt",
            "c8qd",
            "c1qd",
            "ctp",
            "ctG",
            "cbp",
            "ccp",
            "ctap",
            "cmup",
            "ctW",
            "ctZ",
            "ctB",
            "cpl2",
            "c3pl2",
            "cpl3",
            "c3pl3",
            "cpmu",
            "cpta",
            "c3pQ3",
            "cpqMi",
            "cpQM",
            "cpQ",
            "cpt",
            "cptb",
            "cll",
            "cpG",
            "cpGtil",
            "cpd",
            "cW",
            "cB",
        ]
    }
    """ VBS SMEFiT vs FitMarker and BDHLL20 Individual LO, Linear comparison """
    fit_smefit = FitManager(
        f"{path}/VBS_SMEFiT",
        "SNS_VV_VBS_LO_NHO",
        label=r"${\rm SMEFiT\ VV+VBS,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
        is_individual=True,
    )
    fit_fitmarker = FitManager(
        f"{path}/external",
        "FitMaker_INDIV_LO_NHO",
        label=r"${\rm FitMaker\ Top+H+VV,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
        has_posterior=False,
    )
    fit_bdhll = FitManager(
        f"{path}/external",
        "BDHLL20_INDIV_LO_NHO",
        label=r"${\rm BDHLL20\ VV+VH,\ LO\ QCD\ \mathcal{O}(\Lambda^2)}$",
        has_posterior=False,
    )
    report_name = "vv_vbs_individual_bench"

    smefit = Runner(
        f"{report_path}/{report_name}", [fit_smefit, fit_bdhll, fit_fitmarker]
    )
    smefit.run(free_dofs=free_dofs, plot_only=plot_only)


if __name__ == "__main__":

    Individual_bench()
