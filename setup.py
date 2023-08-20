import sys

import setuptools
from setuptools import setup

__version__ = "0.0"

if sys.version_info < (3, 8):
    sys.exit("XGI requires Python 3.8 or later.")

name = "sod"

version = __version__

authors = "Nicholas Landry and Nicole Eikmeier"

author_email = "nicholas.landry@uvm.edu"

description = """Functions and scripts for computing the simpliciality of higher-order datasets."""


def parse_requirements_file(filename):
    with open(filename) as fid:
        requires = [l.strip() for l in fid.readlines() if not l.startswith("#")]
    return requires


install_requires = parse_requirements_file("requirements.txt")

license = "3-Clause BSD license"

setup(
    name=name,
    packages=setuptools.find_packages(),
    version=version,
    author=authors,
    author_email=author_email,
    description=description,
    install_requires=install_requires,
)
