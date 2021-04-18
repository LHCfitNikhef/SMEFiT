# # -*- coding: utf-8 -*-
import numpy as np

def coeff_by_group():
    """
    Lists for SMEFT operator classification:
        4H = 4-heavy
        2L2H = 2-light-2-heavy
        2FB = 2-fermion + boson
        B = purely bosonic
    """  # pylint:disable=line-too-long
    return {
        "4H": [
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
        ],
        "2L2H": [
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
        ],
        "2FB": [
            "ctp",
            "ctG",
            "cbp",
            "ccp",
            "ctap",
            "ctW",
            "ctZ",
            "cbW",
            "cff",
            "cpl1",
            "c3pl1",
            "cpl2",
            "c3pl2",
            "cpl3",
            "c3pl3",
            "cpe",
            "cpmu",
            "cpta",
            "c3pq",
            "c3pQ3",
            "cpqMi",
            "cpQM",
            "cpui",
            "cpdi",
            "cpt",
            "cll",
        ],
        "B": [
            "cpG",
            "cpGtil",
            "cpB",
            "cpBtil",
            "cpW",
            "cpWtil",
            "cpWB",
            "cpWBtil",
            "cpd",
            "cpD",
            "cWWW",
            "cWWWtil",
            "cW",
            "cB",
        ],
    }


def get_conficence_values(dist):
    """
    Get confidence level bounds given the distribution
    """
    cl_vals = {}
    cl_vals["low"] = np.nanpercentile(dist, 16)
    cl_vals["high"] = np.nanpercentile(dist, 84)
    cl_vals["low95"] = np.nanpercentile(dist, 2.5)
    cl_vals["high95"] = np.nanpercentile(dist, 97.5)
    cl_vals["mid"] = np.mean(dist, axis=0)
    cl_vals["error68"] = (cl_vals["high"] - cl_vals["low"]) / 2.0
    cl_vals["error95"] = (cl_vals["high95"] - cl_vals["low95"]) / 2.0
    return cl_vals


def set_double_cl(full_solution, l):

    min_val = min(full_solution)
    max_val = max(full_solution)
    mid = (max_val + min_val) / 2.0

    if l in ["Obp", "Opd"]:
        solution1 = full_solution[full_solution > mid]
        solution2 = full_solution[full_solution < mid]
    else:
        solution1 = full_solution[full_solution < mid]
        solution2 = full_solution[full_solution > mid]

    # First solution
    cl_vals_1 = get_conficence_values(solution1)
    # Second solution
    cl_vals_2 = get_conficence_values(solution2)
    cl_vals_1.update({ "2": cl_vals_2})

    return cl_vals_1, cl_vals_1["mid"] / cl_vals_1["error68"]
    # [ 1./np.sqrt(error68), 1./np.sqrt(error95) ], [ mid / error68 , mid / error95 ]
