# -*- coding: utf-8 -*-
import pathlib
from setuptools import setup, find_packages

import packutil as pack

# write version on the fly - inspired by numpy
MAJOR = 2
MINOR = 0
MICRO = 0

repo_path = pathlib.Path(__file__).absolute().parent


def setup_package():
    # write version
    pack.versions.write_version_py(
        MAJOR,
        MINOR,
        MICRO,
        pack.versions.is_released(repo_path),
        filename="src/smefit_lite/version.py",
    )
    # paste Readme
    with open("README.md", "r") as fh:
        long_description = fh.read()
    # do it
    setup(
        name="smefit_lite",
        version=pack.versions.mkversion(MAJOR, MINOR, MICRO),
        description="A tool to compare SMEFiT posterior distibutions",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Jacob J. Ethier, Jaco Ter Hoeve, Giacomo Magni, Emanuele R. Nocera, Juan Rojo",
        author_email="gmagni@nikhef.nl, enocera@ed.ac.uk, j.rojo@vu.nl",
        url="https://github.com/LHCfitNikhef/SMEFiT",
        package_dir={"": "src/"},
        packages=find_packages("src/"),
        #package_data={"smefit": ["tables/*.yaml"]},
        zip_safe=False,
        classifiers=[
            "Operating System :: Unix",
            "Programming Language :: Python",
            "Programming Language :: Python :: 3",
            "Topic :: Scientific/Engineering",
            "Topic :: Scientific/Engineering :: Physics",
        ],
        install_requires=[
            "rich",
            "matplotlib",
            "pyyaml",
            "numpy",
            "pandas",
        ],
        python_requires=">=3.7",
    )


if __name__ == "__main__":
    setup_package()
