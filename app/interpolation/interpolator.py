from datetime import timedelta
import functools
import logging
import multiprocessing as mp
import sys
from tempfile import mkdtemp
import time

import numpy as np
from scipy.interpolate import interp1d
import joblib

from app.io import readhdf, readgeodata
from app import config

logger = logging.getLogger('ThemisTI')

class EphemerisInterpolator(object):

    emissivities = {0.85: np.s_[:13500],
                    0.90: np.s_[13500:27000],
		    1.0: np.s_[27000:]}

    elevations = {-5: np.s_[: 2700],
                  -2: np.s_[2700:5400],
                  -1: np.s_[5400:8100],
                  6: np.s_[8100:10800],
                  8: np.s_[13500:]}

    time_lookup = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6,
                   12:7, 14:8, 16:9, 24:10, 28:11, 32:12,
                   35:13, 36:14, 40:15, 44:16, 47:17}
    inverse_time_lookup = {v:k for k, v in time_lookup.iteritems()}

    latitude_lookup = dict(zip(range(-90, 95, 5), range(37)))
    inverse_latitude_lookup = {v:k for k, v in latitude_lookup.iteritems()}

    def __init__(self, temperature, ancillarydata, parameters):

	"""
	Attributes
	----------
	temperature : object
		      Geospatial Data object containing temperature data
			and geospatial data
	ancillarydata : dict
			Dict of geospatial data objects with keys the data
			type identifier, e.g. albedo
	startseason : int
		      The first season to read from the hdf5 file
	stopseason : int
		     The final season to read from the hdf5
        data : ndarray
               The interpolated data using all ephemeris interpolation

	"""

	self.temperature = temperature
        self.ancillarydata = ancillarydata
        self.parameters = parameters
	self.hdffile = readhdf.HDFDataSet()
	self.lookuptable = self.hdffile.data

        self.startseason = self.parameters['startseason']
        self.stopseason = self.parameters['stopseason']
        self.season = self.parameters['season']

	self.startlookup = self.lookuptable['season_{}'.format(self.startseason)]
	self.stoplookup = self.lookuptable['season_{}'.format(self.stopseason)]

	self.get_emissivity_offsets()
        self.get_hour_offsets()
        self.get_lat_offsets()

        self.extract_start_data()
        self.extract_stop_data()

        self.interpolateseasons()
        self.interpolatehour()

        self.latitudenodes = [self.inverse_latitude_lookup[k] for k in self.latslice[self.latsort]]

    def extract_start_data(self):
        """
        Extract the start seasons array
        """
        #Double slicing is need because h5py does not support multiple fancy indices
        self.startdata = self.startlookup[:,
                                     self.latslice,
                                     self.startemiss]

        self.startdata = self.startdata[self.hourslice, :, :]
        shape = self.startdata.shape
        nbytes = self.startdata.nbytes
        logger.debug('Seasonal arrays have shape {} and require {} bytes each'.format(shape, nbytes))

    def extract_stop_data(self):
        """
        Extract the stop seasons array
        """
        self.stopdata = self.stoplookup[:,
                                    self.latslice,
                                    self.startemiss]
        self.stopdata = self.stopdata[self.hourslice, :, :]

    def interpolatehour(self):
        """
        Apply a cubic interpolation in hour
        """

        #Get the x node values and check they are monotonically increasing
        hourkeys = self.hourslice[self.hoursort]
        x = [self.inverse_time_lookup[k] for k in hourkeys]
        mask =  np.asarray(self.checkmonotonic(x))
        offset =  max(self.time_lookup.keys()) + 1
        x = np.asarray(x)
        x[~mask] += offset
        x *= 0.5  #Convert from 48 time steps to 24 hours

        f = interp1d(x, self.data.T,
                     kind = config.HOUR_INTERPOLATION,
                     copy=False)
        starttime = self.parameters['starttime']
        newy = starttime.hour + (starttime.minute / 60.0)
        self.data = f(newy)

    def checkmonotonic(self, iterable):
        """
        Check if a given iterable is monotonically increasing.

        Parameters
        ----------
        iterable : iterable
                   Any Python iterable object

        Returns
        -------
        monotonic : list
                    A boolean list of all True if monotonic, or including
                    an inflection point
        """
        monotonic =  [True] + [x < y for x, y in zip(iterable, iterable[1:])]
        return monotonic

    def interpolateseasons(self):
        """
        Linear interpolation to get the seasonal arrays setup
        """

        remainder = self.season - self.startseason
        f1 = 1.0 - remainder
        self.data = (self.startdata * f1) + (self.stopdata * remainder)

    def get_nearest(self, input_value, lookup_table):
        """
        Get the key nearest to the input value from a lookup table.
        """
        keys = lookup_table.keys()
        idx = min(range(len(keys)), key=lambda i: abs(keys[i] - input_value))
        return lookup_table[keys[idx]]

    def get_emissivity_offsets(self):
        """
        Compute the necessary offsets for the provided emissivity
        """
        emissivity = self.ancillarydata['emissivity']
        if isinstance(emissivity, readgeodata.GeoDataSet):
            min_emissivity = emissivty.minimum
            max_emissivity = emissivity.maximum
        else:
            min_emissivity = np.amin(emissivity)
            max_emissivity = np.amax(emissivity)

        logger.debug('Minimum emissivity value is {}'.format(min_emissivity))
        logger.debug('Maximum emissivity value is {}'.format(max_emissivity))

        if min_emissivity in self.emissivities.keys() and max_emissivity in self.emissivities.keys():
            self.startemiss = self.emissivities[min_emissivity]
            self.stopemiss = self.emissivities[max_emissivity]
        else:
            #Case where emissivities are not in the dict keys, need to find the closest
            pass

    def get_hour_offsets(self):
        """
        Compute the hour offsets from the starttime and stoptime
        """
        starttime = self.parameters['starttime']
        stoptime = self.parameters['stoptime']
        timediff = stoptime - starttime
        if timediff > timedelta(minutes = config.TIME_THRESHOLD):
            logger.debug("Time delta is {}.  This is significantly larger than anticipated".format(timediff))
            self.starttime = starttime
        else:
            logger.debug("Time delta is {}.  Using start time as the global time".format(timediff))

        image_time = starttime.hour + (starttime.minute / 60.0)
        middle_idx = int(image_time * 2)
        middle_idx = self.get_nearest(middle_idx, self.time_lookup)
        minhour = middle_idx - 2
        maxhour = middle_idx + 2

        #If the obervations is near 24 hours, wrap
        hourslice = np.arange(minhour, maxhour+1)
        nhours = self.startlookup.shape[0]
        greatermask = np.where(hourslice >= nhours)
        hourslice[greatermask] -= nhours

        lessmask = np.where(hourslice < 0)
        hourslice[lessmask] += self.startlookup.shape[0]

        #Ordered HDF Read to correct interpolation order over the time break.
        self.hoursort = np.argsort(hourslice)
        self.hourslice = hourslice[self.hoursort]
        self.hoursort = np.argsort(self.hoursort)

    def get_lat_offsets(self):
        """
        Given the min and max latitudes, get the full necessary extent.
        TODO: This might be a bad idea if the images is a huge strip over
              a large lat range.
        """
        startlat = self.parameters['startlatitude']
        stoplat = self.parameters['stoplatitude']

        #Given the start and stops,
        start_idx = self.get_nearest(startlat, self.latitude_lookup) - 2
        stop_idx = self.get_nearest(stoplat, self.latitude_lookup) + 2

        #Handle wraps around the poles
        latslice = np.arange(start_idx, stop_idx + 1)
        nlats = self.startlookup.shape[1]
        greatermask = np.where(latslice >= nlats)
        latslice[greatermask] -= nlats
        lessmask = np.where(latslice < 0)
        latslice[lessmask] += self.startlookup.shape[1]

        self.latsort = np.argsort(latslice)
        self.latslice = latslice[self.latsort]
        self.latsort = np.argsort(self.latsort)

        logger.debug('Start latitude node is {}.  Nearest lookup node is {}.'.format(startlat, start_idx))
        logger.debug('Stop latitude node is {}.  Nearest lookup node is {}.'.format(stoplat, stop_idx))

# note that this decorator ignores **kwargs
def memoize(obj):
    cache = obj.cache = {}

    @functools.wraps(obj)
    def memoizer(*args, **kwargs):
        if args not in cache:
            cache[args] = obj(*args, **kwargs)
        return cache[args]
    return memoizer

class ParameterInterpolator(object):

    slopeaz_lookup = {0:np.s_[:540],
                      75:np.s_[540:1080],
                      210:np.s_[1080:1620],
                      285:np.s_[1620:2160],
                      360:np.s_[2160:]}
    #inverse_slopeaz_lookup = {v:k for k, v in slopeaz_lookup.iteritems()}

    slope_lookup = {0:np.s_[:180],
                    30:np.s_[180, 360],
                    60:np.s_[360:]}


    #inverse_slope_lookup = {v:k for k, v in slope_lookup.iteritems()}

    def __init__(self, temperature, td, ed, sd, sz, od, ad,
                 lookuptable, latitudenodes, startpixel):

        self.temperature = temperature
        self.lookup = lookuptable
        self.td = td
        self.ed = ed
        self.sd = sd
        self.sz = sz
        self.ad = ad
        self.od = od
        self.latitudenodes = latitudenodes
        self.start = startpixel
        #memory = self.setupcache()

        self.resultdata = np.empty((self.td.shape[0], self.td.shape[1]), dtype=np.float32)

        self.computelatitudefunction()

    def bruteforce(self):
        """
        Apply a brute force approach to interpolating using a double for loop
        """
        for i in xrange(self.td.shape[0]):
            #Get the latitude at the start of the row, this is used for the entire row

            if i % config.LATITUDE_STEP == 0:
                startlat = i + config.LATITUDE_STEP  #move to the center of the step
                startlat += self.start  #Offset for parallel segmentation
                latitude = self.getlatitude(startlat, self.temperature)
                #Perform the cubic latitude interpolation
                compressedlookup = self.interpolate_latitude(latitude)
                self.elevation_interp_f = interp1d(np.array([-5, -2, -1, 6, 8]),
                                              compressedlookup,
                                              kind=config.ELEVATION_INTERPOLATION,
                                              copy=False,
                                              axis=0)
            for j in xrange(self.td.shape[1]):
                if self.td[i,j] == self.temperature.ndv:
                    #The pixel is no data in the input, propagate to the output
                    self.result = self.temperature.ndv
                else:
                    #Interpolate elevation
                    elevation = self.ed[i,j]
                    new_elevation = self.interpolate_elevation(elevation)

                    #Interpolate Slope Azimuth
                    #slopeaz_f = self.compute_slope_azimuth_function(new_elevation)
                    #new_slopeaz = self.interpolate_slopeaz(slopeaz_f, self.sz[i,j])
                    #print new_slopeaz.shape
                    #Interpolate Slope Azimuth
                    #Interpolate Slope
                    #Interpolate Tau
                    #Interpolate Albedo
                    #Interpolate Inertia

    @memoize
    def interpolate_elevation(self, elevation):
       return self.elevation_interp_f(elevation)

    @memoize
    def interpolate_slopeaz(self, slopeaz_f, slopeaz):
        return slopeaz_f(slopeaz)

    def compute_slope_azimuth_function(self, new_elevation):
        x = self.slopeaz_lookup.keys()
        return interp1d(x, new_elevation,
                                    kind = config.SLOPEAZ_INTERPOLATION,
                                    copy=True,
                                    axis=0)

    def computelatitudefunction(self):
        """
        Given the x, y values, generate the latitude interpolation function
        """
        x = self.latitudenodes
        self.lat_f = interp1d(x, self.lookup,
                              kind = config.LATITUDE_INTERPOLATION,
                              copy = True)

    def interpolate_latitude(self, latitude):
        """
        Given a lookup table, apple the necessary interpolation

        Parameters
        ----------
        latitude : float
                   The latitude at which interpolation is to occurs
        """
        #TODO: The reshape is hard coded, can I get this from the hdf5 file?
        return self.lat_f(latitude).reshape(5, 5, 3, 3, 3, 20)

    def getlatitude(self, y, geoobject):
        """
        Given a row number, compute the current latitude, assuming that
        the column is equal to 0.

        Parameters
        -----------
        y : int
            row identifier

        Returns
        -------
        latitude : float
                   Latitude for the given pixel
        """
        latitude, longitude = geoobject.pixel_to_latlon(0, y)
        return latitude
