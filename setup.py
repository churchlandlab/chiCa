from setuptools import setup, find_packages
from setuptools.command.install import install
import os
import sys
from os.path import join as pjoin
import shutil

            
description = '''chiCa: Processing of miniscope calcium imaging data from the chipmunk task.'''

with open("README.md", "r") as fh:
    long_description = fh.read()
    
setup(
    name = 'chiCa',
    version = '0.1',
    author = 'Lukas Oesch',
    author_email = 'LOesch@mednet.ucla.edu',
    description = (description),
    long_description = long_description,
    long_description_content_type='text/markdown',
    license = 'GPL',
    install_requires = [], #caiman too but it's not through pip I think
    url = "https://github.com/churchlandlab/chiCa",
    packages = find_packages(),
    python_requires='>=3.6'
)