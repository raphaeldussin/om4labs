""" setup for om4labs """
from setuptools import setup, find_packages
import os


is_travis = "TRAVIS" in os.environ

setup(
    name="om4labs",
    version="0.0.1",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("A tool to analyze om4 model output"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/om4labs",
    packages=find_packages(),
    scripts=["scripts/om4labs"],
)
