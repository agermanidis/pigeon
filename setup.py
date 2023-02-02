#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Setup script to create the Pigeon-XT package."""

import io
import os
import sys
from shutil import rmtree

from setuptools import find_packages, setup, Command

NAME = "pigeonXT-jupyter"
DESCRIPTION = "Quickly annotate data in Jupyter notebooks."
URL = "https://github.com/dennisbakhuis/pigeonXT"
EMAIL = "pypi@bakhuis.nu"
AUTHOR = "Dennis Bakhuis"

REQUIRED = [
    "numpy",
    "pandas",
    "ipywidgets",
]

here = os.path.abspath(os.path.dirname(__file__))

with io.open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = "\n" + f.read()

about = {}
with open(os.path.join(here, "pigeonXT", "__version__.py")) as f:
    exec(f.read(), about)


class PublishCommand(Command):
    """Support setup.py publish."""

    description = "Build and publish the package."
    user_options = []

    @staticmethod
    def status(s):
        """Print things in bold."""
        print("\033[1m{0}\033[0m".format(s))

    def initialize_options(self):
        """Initialize options before setup."""
        pass

    def finalize_options(self):
        """Finalize options during setup."""
        pass

    def run(self):
        """Start the package building process."""
        try:
            self.status("Removing previous builds…")
            rmtree(os.path.join(here, "dist"))
        except FileNotFoundError:
            pass

        self.status("Building Source and Wheel (universal) distribution…")
        os.system("{0} setup.py sdist bdist_wheel --universal".format(sys.executable))

        self.status("Uploading the package to PyPi via Twine…")
        os.system("twine upload dist/*")

        sys.exit()


setup(
    name=NAME,
    version=about["__version__"],
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    url=URL,
    packages=["pigeonXT"],
    install_requires=REQUIRED,
    include_package_data=True,
    license="Apache 2.0",
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Framework :: Jupyter",
    ],
    # $ setup.py publish support.
    cmdclass={
        "publish": PublishCommand,
    },
)
