from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="convbnmerge",
    version="0.1.0",
    description="One-clicked merge convolution and batchnorm to one unified convolution",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    url="https://github.com/sksq96/pytorch-summary",
    author="vietnamican",
    author_email="vietnamican@gmail.com",
    packages=["convbnmerge"],
)
