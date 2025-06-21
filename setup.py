from setuptools import setup, find_packages

setup(
    name="py_ssdi",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
    author="Borjan Milinkovic",
    author_email="borjan.milinkovic@gmail.com",
    description="A Python package for Dynamical Independence analysis on Linear State-Space models",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/bmilinkovic/py_ssdi",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
) 