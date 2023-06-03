from setuptools import setup, find_packages

setup(
    name="wind_turbine_power",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        # other dependencies...
    ],
)
