# -*- coding: utf-8 -*-
import numpy as np


def propagate_constraints(config, posterior, is_individual=False, is_results=False):
    """This fuctions construct the posterior ditributions for the
    fixed coefficints

    Parameters
    ----------
        config: dict
            configuration card
        posterior : dict
            posterior distributions
        is_individual: bool, optional
            if the posterior are individual perform a convolution of replicas
        is_results: bool, optional
            use if only results and not posteriors are given

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

        if is_individual:
            # Note this method is statistically equivalent
            # to sum the CL bounds in quadrature in the limit
            # for size going to infinity.
            sigma, mean = 0, 0
            size = int(10e6)
            for (a, post) in zip(rotation, new_post):
                sigma += (a * np.std(post)) ** 2
                mean += a * np.mean(post)
            new_post = np.random.normal(loc=mean, scale=np.sqrt(sigma), size=size)
        elif is_results:
            new_results = dict(zip(new_post[0].keys(), [0] * len(new_post[0].keys())))
            for i, result_dict in enumerate(new_post):
                for key, val in result_dict.items():
                    new_results[key] += np.array(val) * rotation[i]
            new_post = new_results
        else:
            new_post = rotation @ np.array(new_post)

        posterior.update({name: new_post})
