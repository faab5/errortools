from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name='errortools',
    version='0.1.0',
    author="Fabian Jansen",
    author_email="faab5jansen@gmail.com",
    description="Tools for estimating errors",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ]
)
