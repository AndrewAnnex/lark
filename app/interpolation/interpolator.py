from datetime import timedelta
import functools
import logging
import math
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

    slope_lookup = {0:np.s_[:180],
                    30:np.s_[180, 360],
                    60:np.s_[360:]}


    tau_lookup = {0.02:np.s_[:60],
                  0.30:np.s_[6:120],
                  0.62:np.s_[120:]}

    albedo_lookup = {0.08:np.s_[:20],
                     0.22:np.s_[20:40],
                     0.32:np.s_[40:]}

    inertia_values = np.array([24.0, 30.9, 39.9, 51.4, 66.3, 85.5,
                               110.3, 142.2, 183.3, 236.3, 304.7,
                               392.8, 506.5, 653.0, 842.0, 1085.6,
                               1399.7, 1804.7, 2326.8, 3000.0])
    inertia_values_log = np.log(inertia_values)
    inertia_values_ratio = np.empty(len(inertia_values_log) - 1)

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

        self.resultdata = np.empty((self.td.shape[0], self.td.shape[1]), dtype=np.float32)

        self.computelatitudefunction()

    def bruteforce(self):
        """
        Apply a brute force approach to interpolating using a double for loop
        """
        import time
        t1 = time.time()
        for i in xrange(self.td.shape[0]):
            #Get the latitude at the start of the row, this is used for the entire row

            if i % config.LATITUDE_STEP == 0:
                startlat = i + config.LATITUDE_STEP  #move to the center of the step
                startlat += self.start  #Offset for parallel segmentation
                latitude = self.getlatitude(startlat, self.temperature)
                #Perform the cubic latitude interpolation
                compressedlookup = self.interpolate_latitude(latitude)
                self.elevation_interp_f = self.compute_interpolation_function(np.array([-5.0, -2, -1, 6, 8]),
                                            compressedlookup, config.ELEVATION_INTERPOLATION)
            if i % 100 == 0:
                print i, self.td.shape[0], time.time() - t1
            for j in xrange(self.td.shape[1]):
                if self.td[i,j] == self.temperature.ndv:
                    #The pixel is no data in the input, propagate to the output
                    self.result = self.temperature.ndv
                else:
                    #Interpolate elevation
                    elevation = self.ed[i,j]
                    new_elevation = self.interpolate_elevation(elevation)

                    #Interpolate Slope Azimuth
                    slopeaz_f = self.compute_interpolation_function(self.slopeaz_lookup.keys(),
                                                                    new_elevation,
                                                                    config.SLOPEAZ_INTERPOLATION)
                    #slopeaz_f = self.compute_slope_azimuth_function(new_elevation)
                    new_slopeaz = slopeaz_f(self.sz[i,j])
                    #Interpolate Slope
                    slope_f = self.compute_interpolation_function(self.slope_lookup.keys(),
                                                                  new_slopeaz,
                                                                  config.SLOPE_INTERPOLATION)
                    #slope_f = self.compute_slope_function(new_slopeaz)
                    new_slope = slope_f(self.sd[i,j])
                    #Interpolate Tau
                    tau_f = self.compute_interpolation_function(self.tau_lookup.keys(),
                                                                new_slope,
                                                                config.OPACITY_INTERPOLATION)
                    #tau_f = self.compute_tau_function(new_slope)
                    new_tau = tau_f(self.od[i,j])
                    #Interpolate Albedo
                    albedo_f = self.compute_interpolation_function(self.albedo_lookup.keys(),
                                                                   new_tau,
                                                                   config.ALBEDO_INTERPOLATION)
                    #albedo_f = self.compute_albedo_function(new_tau)
                    new_albedo = albedo_f(self.ad[i,j])
                    #Interpolate Inertia
                    self.resultdata[i,j], uncertainty = self.interpolate_ti(self.td[i,j], new_albedo)

    def interpolate_ti(self, temperature, albedo_lookup):
        """
        Interpolate albedo to Thermal Inertia

        Parameters
        ----------
        temperature : float
                      temperature at a given pixel

        albedo_lookup : ndarray
                        vector lookup from the previous interpolations
        """

        maxi = albedo_lookup.argmax()
        mini = albedo_lookup.argmin()
        #Assuming that the model is monnotoic, handle slope of either direction
        if mini < maxi:
            maxi, mini = mini, maxi
        tmax = albedo_lookup[maxi]
        tmin = albedo_lookup[mini]
        monotonic_temps = albedo_lookup[maxi:mini+1]

        num_temps = len(monotonic_temps)
        if np.all(np.diff(monotonic_temps) < 0):
            upperidx = num_temps - np.searchsorted(monotonic_temps[::-1], temperature, side='left')
        else:
            upperidx = np.searchsorted(monotonic_temps, temperature, side='left')

        if upperidx == num_temps:
            #Should return a high temperature error, not 0.0
            return self.inertia_values[-1], 0.0
        elif upperidx == 0:
            return self.inertia_values[0], 0.0
        upperthreshold =  monotonic_temps[upperidx]
        lowerthreshold = monotonic_temps[upperidx - 1]
        fractional_index = upperidx + (temperature - lowerthreshold) / (upperthreshold-lowerthreshold)

        a = max([0.001, abs(albedo_lookup[upperidx] - albedo_lookup[upperidx - 1])])
        uncertainty = min([self.inertia_values_ratio[upperidx - 1]/a, 95.11]) #Hugh's magic uncertainty number

        i = int(fractional_index)
        log_inertia = self.inertia_values_log[maxi]

        rterp = self.inertia_values_log[maxi + i - 1] +\
                (fractional_index - float(i)) *\
                (self.inertia_values_log[maxi+i] -\
                 self.inertia_values_log[maxi + i - 1])

        inertia =  math.exp(rterp)
        return inertia, uncertainty

    @memoize
    def interpolate_elevation(self, elevation):
       return self.elevation_interp_f(elevation)


    def compute_interpolation_function(self, x, inarray,
                                       kind,
                                       bounds_error=False):
        """
        Compute an interpolation function for axis=0 of a
        multi-dimensional array.

        Parameters
        ----------
        x : iterable
            A list or ndarray of x dimension nodes

        inarray : ndarray
                  The input array with dimension 0 equal to
                  len(x).

        kind : {'linear', 'quadratic', 'cubic'}
               The type of interpolation to be performed.

        bounds_error : bool
                       (Default False) Whether or not to alert if the
                       input value falls beyond the interpolation range.

        Returns
        -------

        interp1d : object
                   A scipy interp1d object to perform interpolation.
        """

        return interp1d(x, inarray, kind=kind,copy=True, axis=0,
                        bounds_error = bounds_error)

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
