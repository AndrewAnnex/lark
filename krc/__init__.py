__name__ = 'krc'
__version__ = '0.1.3'
__minimum_isis_version__ = (3,5)

import os
import sys

sys.path.append('/usgs/cpkgs/anaconda3_linux/envs/isis4.0.0/bin')

os.environ['ISISROOT'] = '/usgs/cpkgs/anaconda3_linux/envs/isis4.0.0'
os.environ['ISIS3DATA'] = '/usgs/cpkgs/isis3/data'

# Hard dependency on ISIS3
import pysis
pysis.check_isis_version(*__minimum_isis_version__)
