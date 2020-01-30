''' setup for OM4_Analysis_Labs '''
import setuptools

setuptools.setup(
    name="OM4_Analysis_Labs",
    version="0.0.1",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("A tool to analyze OM4 model output"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/OM4_Analysis_Labs",
    packages=['OM4_Analysis_Labs'],
    scripts=['OM4_Analysis_Labs/annual_bias_1x1deg.py']
)
