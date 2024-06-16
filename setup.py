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

if 'install' in sys.argv or 'develop' in sys.argv:
    from labdatatools.utils import labdata_preferences
    plugins = labdata_preferences['plugins_folder']

    # if the config directory tree doesn't exist, create it
    if not os.path.exists(plugins):
        os.makedirs(plugins)

    # copy every file from given location to the specified ``CONFIG_PATH``
    for fname in os.listdir('analysis'):
        fpath = os.path.join('analysis', fname)
        if not os.path.exists(pjoin(plugins,fname)):
            shutil.copy(fpath, plugins)
        else:
            print('File already exists.')