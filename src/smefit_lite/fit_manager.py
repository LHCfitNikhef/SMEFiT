# -*- coding: utf-8 -*-
import json
import yaml


class FitManager:
    """
    Class to collect all the fits information and
    load the results

    Parameters
    ----------
        path: pathlib.Path
            path to fit location
        name: srt
            fit name
        label: str, optional
            fit label if any otherwise guess it from the name
        has_posterior: bool, optional
            if False load Confidence Level bounds else load the full posterior
        is_individual: bool, optional
            individual fit, needed in case posterior is given
    """

    def __init__(self, path, name, label=None, has_posterior=True, is_individual=False):
        self.path = path
        self.name = name
        self.label = label
        self.has_posterior = has_posterior
        self.is_individual = is_individual
        if self.label is None:
            self.label = r"${\rm %s}$" % name.replace("_", r"\ ")

    def load_configuration(self):
        """Load configuration yaml card

        Returns
        -------
            config: dict
                configuration card
        """
        with open(f"{self.path}/{self.name}/{self.name}.yaml") as f:
            config = yaml.safe_load(f)
        return config

    def load_posterior(self):
        """Load posterior distribution
        Returns
        -------
            posterior : dict
                posterior distibution
        """
        with open(f"{self.path}/{self.name}/posterior.json") as f:
            posterior = json.load(f)
        return posterior

    def load_results(self):
        """Load results yaml card for external
        fits that do not have a full posterior

        Returns
        -------
            results: dict
                available result
        """
        with open(f"{self.path}/{self.name}/results.yaml") as f:
            results = yaml.safe_load(f)
        return results
