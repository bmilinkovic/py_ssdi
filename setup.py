from setuptools import setup, find_packages

setup(
    name="py_ssdi",
    version="0.1.0",
    description="Python implementation of Dynamical Independence for state-space systems",
    author="Borjan Milinkovic",
    author_email="borjan.milinkovic@gmail.com",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "control",
        "networkx",
        "scikit-learn",
        "pandas",
    ],
    python_requires='>=3.8',
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
) 