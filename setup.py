import os
from setuptools import setup, find_packages

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


base_packages = ["numpy", "scipy", "scikit-learn",
                 "pandas", "matplotlib", "jupyter", 
                 "iminuit"]

setup(
    name='errortools',
    version='0.1.0',
    author="Fabian Jansen",
    author_email="faab5jansen@gmail.com",
    description="Tools for estimating errors",
    long_description=read('README.rst'),
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    install_requires=base_packages,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)