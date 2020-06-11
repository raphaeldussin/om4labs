""" setup for om4labs """
from setuptools import setup, find_packages
import os
import subprocess
import sys

platform = 'unknown'
if '--platform' in sys.argv:
    index = sys.argv.index('--platform')
    sys.argv.pop(index)  # Removes the '--platform'
    platform = sys.argv.pop(index)  # get value of arg

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
    package_data={'': ['*.yml']},
    data_files=[
        ('{}/.om4labs'.format(os.environ['HOME']),
         [os.path.join('om4labs/catalogs', 'obs_catalog_gfdl.yml'),
          os.path.join('om4labs/catalogs', 'obs_catalog_gaea.yml'),
          os.path.join('om4labs/catalogs', 'obs_catalog_unknown.yml')
          ])
    ],
)

link_platform = ["ln ", "-s ",
                 os.path.join(os.environ['HOME'], '.om4labs',
                              f'obs_catalog_{platform}.yml'), " ",
                 os.path.join(os.environ['HOME'], '.om4labs',
                              'obs_catalog.yml')]
if not os.path.exists(f"{os.environ['HOME']}/.om4labs/obs_catalog.yml"):
    err = subprocess.check_call(''.join(link_platform), shell=True)
