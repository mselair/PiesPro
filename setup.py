import setuptools
from glob import glob

from setuptools import Command, Extension
import shlex
import subprocess
import os
import re


## get version from file
VERSIONFILE="./PiesPro/__init__.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))



setuptools.setup(
    name="piespro",
    version=verstr,
    license='MFMER',
    url="https://github.com/mselair/PiesPro",

    author="Filip Mivalt",
    author_email="mivalt.filip@mayo.edu",


    description="Python package designed for iEEG analysis and sleep classification.",
    long_description="Python package for EEG sleep classification and analysis. Developed by the laboratory of Bioelectronics Neurophysiology and Engineering - Mayo Clinic",
    long_description_content_type="",

    packages=setuptools.find_packages(),
    include_package_data=True,
    package_data={
        'piespro': ["PiesArt/ArtifactEraser/_configs/*.yaml", "PiesArt/ArtifactEraser/_models/*.pt", "PiesArt/ArtifactBank/*.mat"]
    },
    data_files=[
        ('PiesArt_cfg', glob('PiesArt/ArtifactEraser/_configs/*.yaml')),
        ('PiesArt_mdl', glob('PiesArt/ArtifactEraser/_models/*.pt')),
        ('PiesArt_artifact', glob('PiesArt/ArtifactBank/*.mat')),
    ],


    classifiers=[
        "Development Status :: 3 - Alpha",
        'Intended Audience :: Healthcare Industry',
        'Intended Audience :: Science/Research',
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3.6",
        "Operating System :: OS Independent",
        'Topic :: Scientific/Engineering :: Medical Science Apps.'
    ],
    python_requires='>=3.6',
    install_requires =[
        'numpy',
        'pandas',
        'scipy',
        'tqdm',
        'matplotlib',
        'seaborn',
        'scikit-learn',
        'umap-learn',
        'pymef',
        'mef_tools',
        'pyedflib',
        'h5py',
        'sqlalchemy',
        'PyMySQL',
        'pytz',
        'python-dateutil',
        'pyzmq',
        'sshtunnel',
        'torch',
        'yaml'
    ]
)






