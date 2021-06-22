# -*- coding: utf-8 -*-
import numpy as np


def propagate_constraints(config, posterior):
    """This fuctions construct the posterior ditributions for the
    fixed coefficints

    Parameters
    ----------
        config: dict
            configuration card
        posterior : dict
            posterior distributions

    Returns
    -------
        posterior: dict
            updated posterior distributions
    """

    for name, coeff in config["coefficients"].items():

        if coeff["fixed"] is False or coeff["fixed"] is True:
            continue
        rotation = np.array(coeff["value"])
        new_post = []
        for free_dof in coeff["fixed"]:
            new_post.append(posterior[free_dof])

        new_post = rotation @ np.array(new_post)
        posterior.update({name: new_post})
