<p align="center">
  <a href="https://lhcfitnikhef.github.io/SMEFT/"><img alt="SMEFiT" src=https://github.com/LHCfitNikhef/SMEFT/blob/master/docs/sphinx/_assets/logo.png/>
</a>
</p>

<p align="center">
  <a href="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit/"><img src="https://www.codefactor.io/repository/github/lhcfitnikhef/smefit/badge" alt="CodeFactor" /></a>
</p>


SMEFiT is a Python package for global analyses of particle physics data in the framework of the Standard Model Effective Field Theory (SMEFT). The SMEFT represents a powerful model-independent framework to constrain, identify, and parametrise potential deviations with respect to the predictions of the Standard Model (SM). A particularly attractive feature of the SMEFT is its capability to systematically correlate deviations from the SM between different processes. The full exploitation of the SMEFT potential for indirect New Physics searches from precision measurements requires combining the information provided by the broadest possible dataset, namely carrying out extensive global analysis which is the main purpose of SMEFiT.

In this repository you will find a range of analysis tools that can be used to process the outcome of the SMEFiT analysis, in particular code that takes as input the posterior probability distributions of a given fit and outputs a range of statistical estimators such as CL intervals, residuals, and correlations.

In the near future, the complete SMEFiT fitting framework will be released here open-source with a complete set of documentation and user-friendly analysis examples.

## Installation

To install the ``smefit_lite`` package do:

```bash
git clone https://github.com/LHCfitNikhef/SMEFiT.git
cd SMEFiT
python setup.py install
```

## Running
To run the pogram do:

```bash
python lite_runner.py
```
## Documentation

The documentation is available here: <a href="https://lhcfitnikhef.github.io/SMEFT/"><img alt="Docs" src="https://github.com/LHCfitNikhef/SMEFT/workflows/docs/badge.svg"></a>