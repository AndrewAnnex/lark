#This file contains constant configuration values that are not readable
# from the KRC input files

"""
Time constants
"""
YEAR = 686.9799625
NSEASONS = 80
SEASON_LENGTH = YEAR / float (NSEASONS)
MARTIAN_DAY = 8.5875
RECORD_START_DATE = 144.61074994

#Pressure at elevation zero
MEAN_PRESSURE = 689.700012


"""
The number of minutes over which time is allowed to vary.
Under this threshold time is assumed to be constant across
the entire image.
"""
TIME_THRESHOLD = 10

"""
The parameters below control the interpolation methods used.
Options are: linear, quadratic, cubic, monotonic cubic
"""
HOUR_INTERPOLATION = 'cubic'
LATITUDE_INTERPOLATION = 'cubic' #Needs to be monotonic cubic
ELEVATION_INTERPOLATION = 'cubic'
SLOPEAZ_INTERPOLATION = 'cubic'
SLOPE_INTERPOLATION = 'quadratic'
OPACITY_INTERPOLATION = 'quadratic'
ALBEDO_INTERPOLATION = 'quadratic'
INERTIA_INERPOLATION = 'linear'
"""
The pixel step at which the reference latitude
is recomputed.  This value must be an INTEGER
as we are working in pixel space.
"""
LATITUDE_STEP = 10

"""
This section controls rounding precision.
"""
ELEVATION_ROUNDING = 3


