from setuptools import setup
from setuptools import find_packages

setup(
    name="trainer",
    version="0.1.0",
    install_requires=[
        "tensorflow-gan"
    ],
    packages=find_packages(),
)
