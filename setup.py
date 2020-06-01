''' setup for OM4labs '''
import setuptools

setuptools.setup(
    name="OM4labs",
    version="0.0.1",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("A tool to analyze OM4 model output"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/OM4labs",
    packages=['OM4labs'],
    scripts=['OM4labs/diags/annual_bias_1x1deg/annual_bias_1x1deg.py']
)
