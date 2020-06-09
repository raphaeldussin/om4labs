""" setup for om4labs """
import setuptools

setuptools.setup(
    name="om4labs",
    version="0.0.1",
    author="Raphael Dussin",
    author_email="raphael.dussin@gmail.com",
    description=("A tool to analyze om4 model output"),
    license="GPLv3",
    keywords="",
    url="https://github.com/raphaeldussin/om4labs",
    packages=["om4labs", "om4labs/diags",
              "om4labs/diags/annual_bias_1x1deg"],
    scripts=[
        "scripts/om4labs"
    ],
)
