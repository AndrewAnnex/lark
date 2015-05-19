
import numpy as np
from scipy.interpolate import interp1d

from app import config

slopeaz_lookup = {0:np.s_[:540],
                75:np.s_[540:1080],
                210:np.s_[1080:1620],
                285:np.s_[1620:2160],
                360:np.s_[2160:]}

slope_lookup = {0:np.s_[:180],
              30:np.s_[180, 360],
              60:np.s_[360:]}

def computelatitudefunction(latitudenodes, data):
    """
    Given the x, y values, generate the latitude interpolation function
    """
    x = latitudenodes
    return interp1d(x, data,
                    kind = config.LATITUDE_INTERPOLATION,
                    copy = True)
