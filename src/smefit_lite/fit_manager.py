# -*- coding: utf-8 -*-
import yaml
import json


class FitManager:
    """
    Class to collect all the fits information and
    load the results

    Parameters
    ----------
        path: str
            path to fit location
        name: srt
            fit name
        label: str, optional
            fit label if any otherwise guess it from the name
    """
    def __init__(self, path, name, label=None):
        self.path = path
        self.name = name
        self.label = label
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