from setuptools import setup, find_packages

setup(
    name="wind_turbine_power",
    version="0.1",
    author="Your Name",
    author_email="your.email@example.com",
    description="A package to predict wind turbine power output based on wind speed",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/wind_turbine_power",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scikit-learn",
        "joblib",
        "matplotlib",
        "click"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points='''
        [console_scripts]
        wind_turbine_power=wind_turbine_power.cli:main
    ''',
    python_requires='>=3.7',
)
