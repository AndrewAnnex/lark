import os
from setuptools import setup, find_packages
import krc
#Grab the README.md for the long description
with open('README.md', 'r') as f:
    long_description = f.read()


VERSION = krc.__version__

def setup_package():
    setup(
        name = "krc",
        version = VERSION,
        author = "Jay Laura",
        author_email = "jlaura@usgs.gov",
        description = ("Thermal modeling using KRC thermal inertia lookup tables and THEMIS temperature data."),
        long_description = long_description,
        license = "Public Domain",
        keywords = "geophysical, modeling",
        url = "",
        scripts=['bin/krc_create_changecards', 'bin/krc_submit_changecards',
                 'bin/krc_spawnkrc', 'bin/krc_loadhdf', 'bin/krc_submit_interpolation'],
        packages=find_packages(),
        include_package_data=True,
        zip_safe=False,
        install_requires=['pysis'],
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Utilities",
            "License :: Public Domain",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
    )

if __name__ == '__main__':
    setup_package()
