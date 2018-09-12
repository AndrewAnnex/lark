import functools
import logging
import math
import multiprocessing as mp
import sys
from tempfile import mkdtemp
import time

import numpy as np
from scipy.interpolate import interp1d, PchipInterpolator


from krc import config
from krc.utils import utils

logger = logging.getLogger('ThemisTI')

def compute_interpolation_function(x, inarray,
                                   kind,
                                   bounds_error=True,
                                   axis=0):
    return interp1d(x, inarray, kind=kind,copy=True, axis=axis, bounds_error = bounds_error)

class EphemerisInterpolator(object):

    emissivities = {0.85: np.s_[:16200],
                    0.90: np.s_[16200:32400],
		    1.0: np.s_[32400:]}

    times = np.array([0.0, 1.5, 3, 3.5, 4, 4.5, 5, 6, 7, 8, 12, 14, 16, 17.5, 18, 20, 22, 23.5])
    time_lookup = {v:k for k, v in zip(range(18), times)}
    '''
    time_lookup = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6,
                   12:7, 14:8, 16:9, 24:10, 28:11, 32:12,
                   35:13, 36:14, 40:15, 44:16, 47:17}

    inverse_time_lookup = {v:k for k, v in time_lookup.items()}
    '''
    latitude_lookup = dict(zip(range(-90, 95, 5), range(37)))
    latitudes = np.arange(-90, 95, 5)
    inverse_latitude_lookup = {v:k for k, v in latitude_lookup.items()}

    def __init__(self, temperature, ancillarydata, parameters, hdffile):

        """
        Attributes
        ----------
        temperature   : object
                        Geospatial Data object containing temperature data
                        and geospatial data
        ancillarydata : dict
                        Dict of geospatial data objects with keys the data
                        type identifier, e.g. albedo
        parameters    : dict
                        Additional parameter values
        hdffile       : object
                        A HDF file handle object
        """
        self.temperature = temperature
        self.ancillarydata = ancillarydata
        self.parameters = parameters
        self.lookuptable = hdffile

        self.startseason = self.parameters['startseason']
        self.stopseason = self.parameters['stopseason']
        self.season = self.parameters['season']
        self.startlookup = self.lookuptable['season_{}'.format(self.startseason)]
        self.stoplookup = self.lookuptable['season_{}'.format(self.stopseason)]

    def interpolate_ephemeris(self):
        """
        Helper method that calls the ephemeris interpolation routines
        """
        #Compute the offsets into the lookup tables
        startemiss, stopemiss = self.get_emissivity_offsets()
        hourslice, starttime = self.get_hour_offsets()
        latslice = self.get_lat_offsets()
        
        #Compute the start and stop dates
        startdata = self.extract_season(self.startseason,startemiss,
                                             hourslice, latslice)
        stopdata = self.extract_season(self.stopseason,startemiss,
                                            hourslice, latslice)
        # Interpolate Season
        seasons = [self.startseason, self.stopseason]
        season_f = compute_interpolation_function(seasons, [startdata, stopdata], 'linear')
        data = season_f(self.season)
        #Interpolate time
        self.data = self.interpolatehour(hourslice, starttime, data)
    
    def extract_season(self, season, startemiss, hourslice, latslice):
        lookup = self.lookuptable['season_{}'.format(season)]
        hour_indices = np.argsort(hourslice)
        data = lookup[:,:,startemiss]
         
        # If hourslice is not monotonically increasing, this pulls in the correct order
        data = data[hourslice, :, :]
        data = data[:,latslice, :]
        return data

    def interpolatehour(self, hourslice, starttime, data):
        """
        Apply a cubic interpolation in hour

        """

        times = self.times[hourslice]
        # Hacky solution for boundary crossers
        if times[1] == 0:
            times[0] -= 24
        if times[2] == 0:
            times[0] -= 24
            times[1] -= 24
        elif times[3] == 0:
            times[3] += 24
            times[4] += 24
        elif times[4] == 0:
            times[4] += 24
        f = PchipInterpolator(times,
                              data,
                              extrapolate=False,
                              axis=0)

        return f(starttime)

    def interpolateseasons(self):
        """
        Linear interpolation to get the seasonal arrays setup
        """

        remainder = self.season - self.startseason
        f1 = 1.0 - remainder
        self.data = (self.startdata * f1) + (self.stopdata * remainder)

    def get_emissivity_offsets(self):
        """
        Compute the necessary offsets for the provided emissivity
        """
        emissivity = self.ancillarydata['emissivity']
        min_emissivity = np.amin(emissivity)
        max_emissivity = np.amax(emissivity)

        logger.debug('Minimum emissivity value is {}'.format(min_emissivity))
        logger.debug('Maximum emissivity value is {}'.format(max_emissivity))

        if min_emissivity in self.emissivities.keys() and max_emissivity in self.emissivities.keys():
            startemiss = self.emissivities[min_emissivity]
            stopemiss = self.emissivities[max_emissivity]
        else:
            #Case where emissivities are not in the dict keys, need to find the closest
            #TODO: Emissivity is considered constant at the moment
            startemiss = None
            stopemiss = None
        return startemiss, stopemiss

    def get_hour_offsets(self):
        """
        Compute the hour offsets from the starttime and stoptime
        """
        starttime = self.parameters['startlocaltime']
        stoptime = self.parameters['stoplocaltime']
        timediff = (stoptime - starttime)
        logger.debug("Start time: {} | Stop time: {}".format(starttime, stoptime))
        if timediff > config.TIME_THRESHOLD:
            logger.debug("Time delta is {}.  This is significantly larger than anticipated".format(timediff))
        else:
            logger.debug("Time delta is {}.  Using start time as the global time".format(timediff))

        """
        timediff = (stoptime - starttime).total_seconds()
        logger.debug("Start time: {} | Stop time: {}".format(starttime, stoptime))
        #TODO: How do we want to handle large images with huge time differences?
        if timediff > config.TIME_THRESHOLD:
            logger.debug("Time delta is {}.  This is significantly larger than anticipated".format(timediff))
            starttime = starttime
        else:
            logger.debug("Time delta is {}.  Using start time as the global time".format(timediff))
        """
        #Given the image start time, find the nearest index and set to the middle,
        # then find the adjacent two nodes in both directions to get allow a
        # cubic interpolation.
        #image_time = starttime.hour + starttime.minute / 60.0
        # This grabs the hour that is nearest, but hour is circular
        image_time = starttime
        if abs(image_time - 24) < abs(image_time - 23.5):
            image_time -= 24
        mididx, midhour = utils.getnearest(self.times, image_time)
        logger.debug("Time is {}.  The nearest lookup node is {}".format(image_time, mididx))
        minidx = mididx - 2
        maxidx = mididx + 2

        hourslice = np.arange(minidx, maxidx + 1, dtype=np.int8)

        hourslice[hourslice < 0] += 18

        if hourslice[-1] >= len(self.times):
            #The hour slice needs to be shifted over the time break
            hourslice[hourslice >= len(self.times)] -= len(self.times)
        logger.debug("Using indices {} and start time of {}.".format(hourslice, image_time))
        return hourslice, image_time

    def get_lat_offsets(self):
        """
        Given the min and max latitudes, get the full necessary extent.
        """

        startlat = self.parameters['startlatitude']
        stoplat = self.parameters['stoplatitude']

        #Given the start and stops,
        startidx, startvalue = utils.getnearest(self.latitudes, startlat)
        stopidx, stopvalue = utils.getnearest(self.latitudes, stoplat)
        startidx -= 2
        stopidx += 2
        latslice = np.arange(startidx, stopidx + 1)
        if utils.checkmonotonic(latslice):
            latslice = latslice
        else:
            #TODO: Support pole crossing images
            logger.error('Image is pole crossing, not currently supported.')
            '''
            print "NOT MONOTONIC"
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
            '''
            latslice = None
        logger.debug('Start latitude node is {}.  Nearest lookup node is {}.'.format(startlat, startidx))
        logger.debug('Stop latitude node is {}.  Nearest lookup node is {}.'.format(stoplat, stopidx))
        return latslice

class ParameterInterpolator(object):

    #TODO: All of these offsets should come from the HDF file, not be hard coded here.
    slopeaz_lookup = {0:np.s_[:540],
                      75:np.s_[540:1080],
                      210:np.s_[1080:1620],
                      285:np.s_[1620:2160],
                      360:np.s_[2160:]}

    slope_lookup = {0:np.s_[:180],
                    30:np.s_[180, 360],
                    60:np.s_[360:]}

    elevation_lookup = {-5: np.s_[: 2700],
                  -2: np.s_[2700:5400],
                  -1: np.s_[5400:8100],
                  1 : np.s_[8211:10800],
                  6: np.s_[10800:13500],
                  8: np.s_[16200::]}


    tau_lookup = {0.02:np.s_[:60],
                  0.30:np.s_[60:120],
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

    error_codes = {'elevation_out_of_bounds':1,
                   'monotinicty_constraint_violated':9}

    def __init__(self, temperature, td, ed, sd, sz, od, ad,
                 lookuptable, latitudenodes, startpixel, ndv, r_ndv,
                 reference_name):

        self.temperature = temperature
        self.ndv = ndv
        self.r_ndv = r_ndv
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
        self.log = np.empty((self.td.shape[0], self.td.shape[1]), dtype=np.int8)

        # One of the input datasets is the reference that is clipped to.  In
        # the normal case, this is the temperature data.  In the more complex
        # case this is another one of the input data sets.  Here, we get the
        # the reference dataset in order to perform NDV checks.  The caller
        # passes in the name so that the attribute can be grabbed using getattr.
        self.reference = getattr(self, reference_name)

    '''def convert_elevation_to_pressure(self):
        """
        Convert lookup table and elevation dataset from km to pascals
        """
        new_elev = {}
        for k, v in self.elevation_lookup.items():
            new_key = utils.convert_pressure(k)
            new_elev[new_key] = elevation_lookup[k]

        self.elevation_lookup = new_elev
        self.ed = utils.convert_pressure(self.ed)'''

    def apply_interpolation_func(self, x, func):
        """
        Parameters
        -----------
        x : array
            The input array to be interpolated

        func : object
               The interpolation function to apply

        Returns
        -------
        """
        return func(x)

    def bruteforce(self):
        """
        Apply a brute force approach to interpolating using a double for loop
        """
        import time
        t1 = time.time()
        for i in range(self.td.shape[0]):
            #Get the latitude at the start of the row, this is used for the entire row

            if i % config.LATITUDE_STEP == 0:
                startlat = i + config.LATITUDE_STEP  #move to the center of the step
                startlat += self.start  #Offset for parallel segmentation

                # This is the latitude at the center of the tile defined by
                # the image width, and the latitude_step
                x = int(self.td.shape[1] / 2)
                y = int((startlat + config.LATITUDE_STEP) / 2)
                latitude, _ = self.temperature.pixel_to_latlon(x,y)

                lat_f = PchipInterpolator(self.latitudenodes, self.lookup, extrapolate=False, axis=0)
                #The reshape corresponds to the dimensions of the OLAP cube
                # 5 elevations, 5 slope azimuths, 3 slopes, 3 opacities, 3 albedos, and finally 20 TI
                data = lat_f(latitude)
                compressedlookup = data.reshape(6,5,3,3,3,20)
                # Compute the PChip interpolation function for elevation
                elevation_interp_f = PchipInterpolator(np.array([-5.0, -2.0, -1.0, 1.0, 6.0, 8.0]), compressedlookup, extrapolate=False, axis=0)
            
            for j in range(self.td.shape[1]):
                # Each interpolation is composed in 2 parts.
                # 1. The interpolation function is computed.
                # 2. The interpolation function is applied.
                #print(self.reference[i,j], self.r_ndv)
                # If either the reference or the input THEMIS have no data
                if (self.td[i,j] == self.ndv) or (self.reference[i,j] == self.r_ndv):
                    #The pixel is no data in the input, propagate to the output
                    self.resultdata[i,j] = self.ndv
                    continue

                #Interpolate elevation
                try:
                    new_elevation = elevation_interp_f(self.ed[i,j])
                except:
                    # The elevation is bad.
                    self.resultdata[i,j] = self.ndv
                    self.log[i,j] = self.error_codes['elevation_out_of_bounds']
                    continue
                #Interpolate Slope Azimuth
                slopeaz_f = self.compute_interpolation_function(sorted(self.slopeaz_lookup.keys()),
                                                                new_elevation,
                                                                config.SLOPEAZ_INTERPOLATION)
                new_slopeaz = slopeaz_f(self.sz[i,j])
                #Interpolate Slope
                slope_f = self.compute_interpolation_function(sorted(self.slope_lookup.keys()),
                                                              new_slopeaz,
                                                              config.SLOPE_INTERPOLATION)
                capped_slope = self.sd[i,j]
                if capped_slope > 60.0:
                    capped_slope = 60.0
                new_slope = slope_f(capped_slope)
                # I am having problems here with pulling TAU properly - check montabone!
                #Interpolate Tau
                tau_f = PchipInterpolator(sorted(self.tau_lookup.keys()),
                                                            new_slope,
                                                            extrapolate=False,
                                                            axis=0)
                new_tau = tau_f(self.od[i,j])
                #Interpolate Albedo
                albedo_f = self.compute_interpolation_function(sorted(self.albedo_lookup.keys()),
                                                               new_tau,
                                                               config.ALBEDO_INTERPOLATION)
                new_albedo = albedo_f(self.ad[i,j])
                #Interpolate Inertia
                self.resultdata[i,j] = self.extract_monotonic(self.td[i,j],
                                                              new_albedo)

    def extract_monotonic(self, temp, new_albedo):
        """
        Given an input temperature and vector of temperatures,
        apply a linear interpolation between the two nearest nodes.
        """
        # Check to see how many other values are equal to 'nearest'
        # if temps are [146,146,146,...] the nearest is ambiguous
        nearest = min(new_albedo, key=lambda x: abs(x-temp))
        idx = [i for i, j in enumerate(new_albedo) if j == nearest]
        if not idx:
            logger.debug('idx error: {} {} {} {} '.format(idx, nearest, temp, new_albedo))
            return 0
        elif len(idx) > 1:
            logger.debug('Unable to determine which nodes to interpolate between for pixel.')
            return 0
       
        idx = idx[0]
        # Temperature is on a node
        if temp == new_albedo[idx]:
            return self.inertia_values[idx]
        # Temp is too high
        elif temp > max(new_albedo):
            return 3000
        elif temp < min(new_albedo):
            return 0
        elif temp < nearest:
            nodes = [idx - 1, idx]
        elif temp > nearest:
            nodes = [idx, idx + 1]
        
        # Check that the temperature nodes are valid
        temp_vals = new_albedo[nodes]
        if not (temp_vals[0] <= temp <= temp_vals[1]):
            if temp_vals[1] < temp:
                # The upper bound is under the temp
                while temp_vals[1] < temp:
                    if nodes[1] > len(new_albedo):
                        logger.debug('Bad upper bound', temp_vals, temp)
                        return 0
                    nodes[1] += 1
                    temp_vals[1] = new_albedo[nodes[1]]
            elif temp_vals[0] > temp:
                # The lower bound is above the temp
                while temp_vals[0] > temp:
                    if nodes[0] < 0:
                        logger.debug('Band lower bound', temp_vals, temp)
                        return 0
                nodes[0] -= 1
                temp_vals[0] = new_albedo[nodes[0]]

        # Linear in log space, so grab the nodes and values
        inertia_nodes = self.inertia_values_log[nodes]
            
        inertia_f = self.compute_interpolation_function(temp_vals, inertia_nodes, 'linear')
        res = np.exp(self.apply_interpolation_func(temp, inertia_f))
        return res


    @utils.memoize
    def interpolate_elevation(self, elevation):
       return self.elevation_interp_f(elevation)

    def compute_interpolation_function(self, x, inarray,
                                       kind,
                                       bounds_error=True,
                                       axis=0):
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
                       (Default True) Whether or not to alert if the
                       input value falls beyond the interpolation range.

        axis : int
               The axis along which the interpolation is applied.

        Returns
        -------

        interp1d : object
                   A scipy interp1d object to perform interpolation.

        """

        return interp1d(x, inarray, kind=kind,copy=True, axis=axis,
                        bounds_error=bounds_error)
