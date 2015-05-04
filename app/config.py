#This file contains constant configuration values that are not readable
# from the KRC input files

YEAR = 686.9799625
NSEASONS = 80
SEASON_LENGTH = YEAR / float (NSEASONS)
MARTIAN_DAY = 8.5875
#This is the start date for season 0
RECORD_START_DATE = 144.61074994

"""
The number of minutes over which time is allowed to vary.
Under this threshold time is assumed to be constant across
the entire image.
"""
TIME_THRESHOLD = 10
