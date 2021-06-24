# # -*- coding: utf-8 -*-
import numpy as np


def latex_coeff():
    """
    Dictionary SMEFT operators with latex names
    """
    return {
        # 4H
        "cQQ1": r"$c_{QQ}^{1}$",
        "cQQ8": r"$c_{QQ}^{8}$",
        "cQt1": r"$c_{Qt}^{1}$",
        "cQt8": r"$c_{Qt}^{8}$",
        "cQb1": r"$c_{Qb}^{1}$",
        "cQb8": r"$c_{Qb}^{8}$",
        "ctt1": r"$c_{tt}^{1}$",
        "ctb1": r"$c_{tb}^{1}$",
        "ctb8": r"$c_{tb}^{8}$",
        "cQtQb1": r"$c_{QtQb}^{1}$",
        "cQtQb8": r"$c_{QtQb}^{8}$",
        # "2L2H"
        "c81qq": r"$c_{qq}^{1,8}$",
        "c11qq": r"$c_{qq}^{1,1}$",
        "c83qq": r"$c_{qq}^{8,3}$",
        "c13qq": r"$c_{qq}^{1,3}$",
        "c8qt": r"$c_{qt}^{8}$",
        "c1qt": r"$c_{qt}^{1}$",
        "c8ut": r"$c_{ut}^{8}$",
        "c1ut": r"$c_{ut}^{1}$",
        "c8qu": r"$c_{qu}^{8}$",
        "c1qu": r"$c_{qu}^{1}$",
        "c8dt": r"$c_{dt}^{8}$",
        "c1dt": r"$c_{dt}^{1}$",
        "c8qd": r"$c_{qd}^{8}$",
        "c1qd": r"$c_{qd}^{1}$",
        # "2FB"
        "ctp": r"$c_{t \varphi}$",
        "ctG": r"$c_{tG}$",
        "cbp": r"$c_{b \varphi}$",
        "ccp": r"$c_{c \varphi}$",
        "ctap": r"$c_{\tau \varphi}$",
        "ctW": r"$c_{tW}$",
        "ctZ": r"$c_{tZ}$",  # Non Warsaw
        "cbW": r"$c_{bW}$",
        "cff": r"$c_{ff}$",
        "cpl1": r"$c_{\varphi l_1}$",
        "c3pl1": r"$c_{\varphi l_1}^{3}$",
        "cpl2": r"$c_{\varphi l_2}$",
        "c3pl2": r"$c_{\varphi l_2}^{3}$",
        "cpl3": r"$c_{\varphi l_3}$",
        "c3pl3": r"$c_{\varphi l_3}^{3}$",
        "cpe": r"$c_{\varphi e}$",
        "cpmu": r"$c_{\varphi \mu}$",
        "cpta": r"$c_{\varphi \tau}$",
        "c3pq": r"$c_{\varphi q}^{3}$",
        "c3pQ3": r"$c_{\varphi Q}^{3}$",
        "cpqMi": r"$c_{\varphi q}^{(-)}$",  # Non Warsaw
        "cpQM": r"$c_{\varphi Q}^{(-)}$",  # Non Warsaw
        "cpui": r"$c_{\varphi u}$",
        "cpdi": r"$c_{\varphi d}$",
        "cpt": r"$c_{\varphi t}$",
        "cptb": r"$c_{\varphi tb}$",  # Non SMEFiT2.0
        "cll": r"$c_{ll}$",
        # "B"
        "cpG": r"$c_{\varphi G}$",
        "cpGtil": r"$c_{\widetilde{\varphi G}}$",  # Non SMEFiT2.0
        "cpB": r"$c_{\varphi B}$",
        "cpBtil": r"$c_{\widetilde{\varphi B}}$",  # Non SMEFiT2.0
        "cpW": r"$c_{\varphi W}$",
        "cpWtil": r"$c_{\widetilde{\varphi W}}$",  # Non SMEFiT2.0
        "cpWB": r"$c_{\varphi WB}$",
        "cpWBtil": r"$c_{\widetilde{\varphi WB}}$",  # Non SMEFiT2.0
        "cpd": r"$c_{\varphi d}$",
        "cpD": r"$c_{\varphi D}$",
        "cWWW": r"$c_{WWW}$",
        "cWWWtil": r"$c_{\widetilde{WWW}}$",
        "cW": r"$c_{W}$",  # Non Warsaw
        "cB": r"$c_{B}$",  # Non Warsaw
    }


def coeff_by_group():
    """
    Lists for SMEFT operator classification:
        4H = 4-heavy
        2L2H = 2-light-2-heavy
        2FB = 2-fermion + boson
        B = purely bosonic
    """
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
            "cptb",
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


def get_confidence_values(dist):
    """
    Get confidence level bounds given the distribution
    """
    cl_vals = {}
    cl_vals["low68"] = np.nanpercentile(dist, 16)
    cl_vals["high68"] = np.nanpercentile(dist, 84)
    cl_vals["low95"] = np.nanpercentile(dist, 2.5)
    cl_vals["high95"] = np.nanpercentile(dist, 97.5)
    cl_vals["mid"] = np.mean(dist, axis=0)
    cl_vals["error68"] = (cl_vals["high68"] - cl_vals["low68"]) / 2.0
    cl_vals["error95"] = (cl_vals["high95"] - cl_vals["low95"]) / 2.0
    cl_vals["cl68"] = np.array(
        [cl_vals["mid"] - cl_vals["low68"], cl_vals["high68"] - cl_vals["mid"]]
    )
    cl_vals["cl95"] = np.array(
        [cl_vals["mid"] - cl_vals["low95"], cl_vals["high95"] - cl_vals["mid"]]
    )
    return cl_vals


def split_solution(full_solution):

    min_val = min(full_solution)
    max_val = max(full_solution)
    mid = (max_val + min_val) / 2.0

    # solution 1 should be closer to 0
    solution1 = full_solution[full_solution < mid]
    solution2 = full_solution[full_solution > mid]

    if min(abs(solution2)) < min(abs(solution1)):
        solution1, solution2 = solution2, solution1

    return solution1, solution2


def get_double_cls(full_solution):
    solution1, solution2 = split_solution(full_solution)
    # First solution
    cl_vals_1 = get_confidence_values(solution1)
    # Second solution
    cl_vals_2 = get_confidence_values(solution2)

    return cl_vals_1, cl_vals_2


def compute_confidence_level(posterior, coeff_list, disjointed_list=None):
    """
    Compute central value, 95 % and 68 % confidence levels and store the result in a dictionary
    given a posterior distribution
    Parameters
    ----------
        posterior : dict
            posterior distibutions per coefficent
        coeff_list : list
            coefficients list for which the bounds are computed
        disjointed_list: list, optional
            list of coefficients with double solutions

    Returns
    -------
        bounds: dict
            confidence level bounds per coefficient
            Note: double solutions are appended under "2"
    """

    disjointed_list = disjointed_list or []
    bounds = {}
    for l in coeff_list:
        if l not in posterior:
            posterior[l] = np.array(np.zeros(len(list(posterior.values())[0])))
        posterior[l] = np.array(posterior[l])
        cl_vals = {}
        # double soultion
        if l in disjointed_list:
            cl_vals1, cl_vals2 = get_double_cls(posterior[l])
            bounds.update({l: cl_vals1})
            bounds.update({f"{l}_2": cl_vals2})
        # single solution
        else:
            cl_vals = get_confidence_values(posterior[l])
            bounds.update({l: cl_vals})

    return bounds


def get_bounds_from_cl(extreme, cl):
    """Get the confidence level, mean and error from bounds

    Parameters
    ----------
        extereme: np.narray
            list of bounds: min, max
        cl: int
            confidence level value
    Returns
    -------
        cl_vals: dict
            Confidence level dictionary
    """
    cl_vals = {
        f"low{cl}": extreme[0],
        f"high{cl}": extreme[1],
        "mid": np.mean(extreme),
    }
    cl_vals.update(
        {
            f"error{cl}": (cl_vals[f"high{cl}"] - cl_vals[f"low{cl}"]) / 2.0,
            f"cl{cl}": np.array(
                [
                    cl_vals["mid"] - cl_vals[f"low{cl}"],
                    cl_vals[f"high{cl}"] - cl_vals["mid"],
                ]
            ),
        }
    )
    return cl_vals


def load_confidence_levels(result, coeff_list, disjointed_list=None):
    """
    Load central value, 95 % and 68 % confidence levels and store the result in a dictionary
    given some inputs results
    Parameters
    ----------
        result : dict
            results per coefficients, at lest '95cl' must be inluded
        coeff_list : list
            coefficients list for which the bounds are computed
        disjointed_list: list, optional
            list of coefficients with double solutions

    Returns
    -------
        bounds: dict
            confidence level bounds per coefficient
            Note: double solutions are appended under "2"
    """

    disjointed_list = disjointed_list or []
    bounds = {}
    for l in coeff_list:
        if l not in result:
            result[l] = {"95cl": np.zeros(2)}
        cl_vals = get_bounds_from_cl(result[l]["95cl"], 95)
        if "68cl" in result[l]:
            cl_vals.update(get_bounds_from_cl(result[l]["68cl"], 68))
        bounds.update({l: cl_vals})
    return bounds
