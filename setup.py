from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="convbnmerge",
    version="0.1.4",
    description="One-clicked merge convolution and batchnorm to one unified convolution",
    long_description=long_description,      # Long description read from the the readme file
    long_description_content_type="text/markdown",
    url="https://github.com/vietnamican/conv-bn-merge",
    author="vietnamican",
    author_email="vietnamican@gmail.com",
    packages=["convbnmerge"],
)
