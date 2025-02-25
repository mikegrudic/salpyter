"""Setup file for salpyter package"""

import os
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

thelibFolder = os.path.dirname(os.path.realpath(__file__))
requirementPath = thelibFolder + "/requirements.txt"
install_requires = []
if os.path.isfile(requirementPath):
    with open(requirementPath) as f:
        install_requires = f.read().splitlines()

setuptools.setup(
    name="salpyter",
    version="0.1",
    author="Mike Grudic",
    author_email="mike.grudic@gmail.com",
    description="Routines for stellar initial mass function computations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    project_urls={
        "Bug Tracker": "https://github.com/mikegrudic/salpyter",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=install_requires,
)
