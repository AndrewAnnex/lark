from datetime import timedelta
import functools
import logging
import math
import multiprocessing as mp
import sys
from tempfile import mkdtemp
import time

import numpy as np
from plio import io_gdal
from scipy.interpolate import interp1d, PchipInterpolator


from krc import config
from krc.utils import utils

logger = logging.getLogger('ThemisTI')

class EphemerisInterpolator(object):

    times = np.arange(24)
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
        self.get_hour_offsets()
        self.get_lat_offsets()
        
        #Compute the start and stop dates
        self.extract_start_data()
        self.extract_stop_data()

        #Interpolate season and time
        self.interpolateseasons()
        self.interpolatehour()
    
    def extract_start_data(self, data='start'):
        """
        Extract the start seasons array
        
        Parameters
        ----------
        data : {start, stop}
               Whether the extraction is the start array or the stop array 
        """
        hour_indices = np.argsort(self.hourslice) 
        hour = self.hourslice[hour_indices]
        #Double slicing is need because h5py does not support multiple fancy indices
        self.startdata = self.startlookup[:,
                                         hour,
                                         :]
        self.startdata = self.startdata[:, :, self.latslice]

        #Hour indices may not be monotonic, so reshape the start data to ensure monotinicity
        idx = utils.check_change_by(self.hourslice, piecewise=True)
        if False in idx:
            idx = idx.index(False)
            h = np.arange(5)
            monotonic_hour = np.hstack((h[idx:], h[:idx]))
            self.startdata = self.startdata[:,monotonic_hour,:]

        shape = self.startdata.shape
        nbytes = self.startdata.nbytes
        logger.debug('Seasonal arrays have shape {} and require {} bytes each'.format(shape, nbytes))

    def extract_stop_data(self):
        """
        Extract the stop seasons array
        """
        hour_indices = np.argsort(self.hourslice) 
        hour = self.hourslice[hour_indices]
        #Double slicing is need because h5py does not support multiple fancy indices
        self.stopdata = self.stoplookup[:,
                                         hour,
                                         :]
        self.stopdata = self.stopdata[:, :, self.latslice]

        #Hour indices may not be monotonic, so reshape the start data to ensure monotinicity
        idx = utils.check_change_by(self.hourslice, piecewise=True)
        if False in idx:
            idx = idx.index(False)
            h = np.arange(5)
            monotonic_hour = np.hstack((h[idx:], h[:idx]))
            self.stopdata = self.stopdata[:,monotonic_hour,:]

        shape = self.stopdata.shape
        nbytes = self.stopdata.nbytes
        logger.debug('Seasonal arrays have shape {} and require {} bytes each'.format(shape, nbytes))

    def interpolatehour(self):
        """
        Apply a cubic interpolation in hour
        """
        times = self.times[self.hourslice]
        if utils.checkmonotonic(times):
            hournodes = self.hourslice
        else:
            #Reorder the nodes and data such that nodes are monotonically increasing
            reorder = np.argsort(times)
            hournodes = self.hourslice[reorder]
            self.data = self.data[:, reorder, :]
        f = interp1d(self.times[hournodes],
                     self.data,
                     kind = config.HOUR_INTERPOLATION,
                     copy=False,
                     axis=1)
        starttime = self.parameters['starttime']
        #newy = starttime.hour + (starttime.minute / 60.0)
        self.data = f(starttime)

    def interpolateseasons(self):
        """
        Linear interpolation to get the seasonal arrays setup
        """

        remainder = self.season - self.startseason
        f1 = 1.0 - remainder
        self.data = (self.startdata * f1) + (self.stopdata * remainder)
    
    def get_hour_offsets(self):
        """
        Compute the hour offsets from the starttime and stoptime
        """
        starttime = self.parameters['starttime']
        stoptime = self.parameters['stoptime']
        timediff = stoptime - starttime
        logger.debug("Start time: {} | Stop time: {}".format(starttime, stoptime))
        #TODO: How do we want to handle large images with huge time differences?
        if timediff > config.TIME_THRESHOLD:
            logger.debug("Time delta is {}.  This is significantly larger than anticipated".format(timediff))
            self.starttime = starttime
        else:
            logger.debug("Time delta is {}.  Using start time as the global time".format(timediff))

        #Given the image start time, find the nearest index and set to the middle,
        # then find the adjacent two nodes in both directions to get allow a
        # cubic interpolation.
        #image_time = starttime.hour + (starttime.minute / 60.0)
        image_time = starttime
        mididx, midhour = utils.getnearest(self.times, image_time)
        logger.debug("Time is {}.  The nearest lookup node is {}".format(midhour, mididx))
        minidx = mididx - 2
        maxidx = mididx + 2
        hourslice = [minidx, mididx, maxidx]
        self.hourslice = np.arange(minidx, maxidx + 1, dtype=np.int8)
        if self.hourslice[-1] >= len(self.times):
            #The hour slice needs to be shifted over the time break
            self.hourslice[self.hourslice >= len(self.times)] -= len(self.times)
        logger.debug("Hour offsets from the time are: {}".format(self.hourslice))
        #self.hourslice = self.times[hourslice]

    def get_lat_offsets(self):
        """
        Given the min and max latitudes, get the full necessary extent.
        #TODO: This might be a bad idea if the images is a huge strip over
              a large lat range.
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
            self.latslice = latslice
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
        logger.debug('Start latitude node is {}.  Nearest lookup node is {}.'.format(startlat, startidx))
        logger.debug('Stop latitude node is {}.  Nearest lookup node is {}.'.format(stoplat, stopidx))


class ParameterInterpolator(object):

    #TODO: All of these offsets should come from the HDF file, not be hard coded here.
    elevation_lookup = {-4.0: np.s_[: 180],
                         0.0: np.s_[180:360],
                         7.0: np.s_[360:]}

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

    def __init__(self, temperature, td, ed, od, ad,
                 lookuptable, latitudenodes, startpixel):

        self.temperature = temperature
        self.ndv = self.temperature.no_data_value
        self.lookup = lookuptable
        self.td = td
        self.ed = ed
        self.ad = ad
        self.od = od
        self.latitudenodes = latitudenodes
        self.start = startpixel
        self.resultdata = np.empty((self.td.shape[0], self.td.shape[1]), dtype=np.float32)
        self.log = np.empty((self.td.shape[0], self.td.shape[1]), dtype=np.int8)
        #self.computelatitudefunction()

    def convert_elevation_to_pressure(self):
        """
        Convert lookup table and elevation dataset from km to pascals
        """
        new_elev = {}
        for k, v in self.elevation_lookup.items():
            new_key = utils.convert_pressure(k)
            new_elev[new_key] = elevation_lookup[k]

        self.elevation_lookup = new_elev
        self.ed = utils.convert_pressure(self.ed)

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
                latitude = self.getlatitude(startlat, self.temperature)
                #Perform the cubic latitude interpolation
                compressedlookup = self.apply_interpolation_func(latitude, self.lat_f).reshape(3,3,3,20)
                elevation_interp_f = self.compute_interpolation_function(np.array([-4.379, 0.0, 7.486]),
                                            compressedlookup, 'linear')
            for j in range(self.td.shape[1]):
                if self.td[i,j] == self.ndv:
                    #The pixel is no data in the input, propagate to the output
                    self.resultdata[i,j] = self.ndv
                else:
                    #Interpolate elevation
                    try:
                        new_elevation = self.apply_interpolation_func(self.ed[i,j], elevation_interp_f)
                    except:
                        self.resultdata[i,j] = self.ndv
                        self.log[i,j] = self.error_codes['elevation_out_of_bounds'] 
                        continue

                    #Interpolate Tau
                    tau_f = self.compute_interpolation_function(sorted(self.tau_lookup.keys()),
                                                                new_elevation,
                                                                config.OPACITY_INTERPOLATION)
                    new_tau = self.apply_interpolation_func(self.od[i,j], tau_f)
                    
                    #Interpolate Albedo
                    albedo_f = self.compute_interpolation_function(sorted(self.albedo_lookup.keys()),
                                                                   new_tau,
                                                                   config.ALBEDO_INTERPOLATION)
                    new_albedo = self.apply_interpolation_func(self.ad[i,j], albedo_f)
                    
                    #Interpolate Inertia

                    nodes, inertia_values = self.extract_monotonic(new_albedo, self.td[i,j])
                    if nodes[0] == nodes[1]:
                        #The result is on a node.
                        #logger.error('Monotonic temperature check yielded a single node.  Temp: {} | inertia: {}'.format(monotonic_temps, inertia_values))
                        self.resultdata[i,j] = np.exp(inertia_values[0])
                    else:
                        inertia_f = self.compute_interpolation_function(nodes,
                                                    inertia_values,
                                                    'linear')
                        '''
                        #Bound check the input temp and the lookup table    
                        if self.td[i,j] < monotonic_temps[0]:
                            self.resultdata[i,j] = np.exp(inertia_values[0])
                        elif self.td[i,j] > monotonic_temps[-1]:
                            self.resultdata[i,j] = np.exp(inertia_values[-1])
                        else:
                        '''
                        try:
                            self.resultdata[i,j] = np.exp(self.apply_interpolation_func(self.td[i,j], inertia_f))
                        except:
                            logger.debug('Error computing final thermal inertia',self.td[i,j], nodes, new_albedo)
                    #self.resultdata[i,j], uncertainty = self.interpolate_ti(self.td[i,j], new_albedo)

    def extract_monotonic(self, new_albedo, temperature):
        """
        Check that temperatures are monotonically increasing and clip those values which are not
        """
        nearest_idx, nearest_value = utils.getnearest(new_albedo, temperature)
        
        #Bounds check:
        if (temperature > new_albedo).all():
            return [0,0], [self.inertia_values_log[-1]]
        elif (temperature < new_albedo).all():
            return [0,0], [self.inertia_values_log[0]]
        else:
            ilo = np.argmin(new_albedo)
            ihi = np.argmax(new_albedo)
            if ilo > ihi:
                ihi, ilo = ilo, ihi
            return [new_albedo[ilo], new_albedo[ihi]], [self.inertia_values_log[ilo], self.inertia_values_log[ihi]]
        #Temperate is valid within the available nodes
        if nearest_value >= temperature > new_albedo[nearest_idx - 1]:
            nextidx = nearest_idx - 1
        elif nearest_value <= temperature < new_albedo[nearest_idx - 1]:
            next_idx = nearest_idx - 1
        elif nearest_value <= temperature < new_albedo[nearest_idx + 1]:
            nextidx = nearest_idx + 1
        elif nearest_value >= temperature > new_albedo[nearest_idx +1]:
            nextidx = nearest_idx + 1
        else:
            nextidx = nearest_idx
        indices = [nearest_idx, nextidx]
        if not utils.checkmonotonic(new_albedo[indices]):
            indices = indices[::-1]
        nodes = new_albedo[indices]
        #The nearest is not necesarily going to give a good range
        na = np.copy(new_albedo)
        while not (nodes[0] <= temperature <= nodes[1]):
            #Modify the vector by masking the previous 'nearest' and searching again
            nmask = np.where(na == np.inf)[0]
            if len(nmask) == 0:
                na[nearest_idx] = np.inf
            else:
                try:
                  na[nearest_idx] = np.inf
                except:
                    logger.debug('Monotonic search issues',temperature, new_albedo)
            #Get the new nearest
            nearest_idx, nearest_value = utils.getnearest(na, temperature)
            
            #Get an adjacent node and check to see if it fulfills the bounding requirement
            if nearest_value - temperature > 0:
                nextidx = nearest_idx - 1
            elif nearest_value - temperature < 0:
                nextidx = nearest_idx + 1
            else:
                nextidx = nearest_idx

            indices = [nearest_idx, nextidx]

            if not utils.checkmonotonic(new_albedo[indices]):
                indices = indices[::-1]
            nodes = new_albedo[indices]
        
        #Using the good indices, return the inertias as well.
        inertia_values = self.inertia_values_log[indices]

        return nodes, inertia_values
        '''        
        #TODO: The logic here is C- style, this assumes the vector is largely monotonic...but we have not idea.
        mask = utils.checkmonotonic(new_albedo, piecewise=True)
        indices = [i for i, x in enumerate(mask) if x == False]
        if len(indices) >= 17:
            #Assume that the array is reversed and try again
            new_albedo = new_albedo[::-1]
            mask = utils.checkmonotonic(new_albedo, piecewise=True)
            indices = [i for i,x in enumerate(mask) if x == True]
            if not indices:
                finalidx = 0
            else:
                finalidx = indices[-1]
        else:
            if not indices:
                finalidx = len(new_albedo)
            else:
                finalidx = indices[-1]
        monotonic_albedo = new_albedo[:finalidx]
        inertia_values = self.inertia_values_log[:finalidx]

        return monotonic_albedo, inertia_values, finalidx
        '''

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
            p
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
                (self.inertia_values_log[maxi + i] -\
                 self.inertia_values_log[maxi + i - 1])

        inertia =  math.exp(rterp)
        return inertia, uncertainty

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
                       (Default False) Whether or not to alert if the
                       input value falls beyond the interpolation range.

        axis : int
               The axis along which the interpolation is applied.

        Returns
        -------

        interp1d : object
                   A scipy interp1d object to perform interpolation.
    
        """
        return interp1d(x, inarray, kind=kind, copy=True, axis=axis,
                        bounds_error = bounds_error)
   
    
    def compute_latitude_function(self):
        """
        Given the x, y values, generate the latitude interpolation function
        """
        x = np.array(self.latitudenodes)
        self.lat_f = PchipInterpolator(x, self.lookup, extrapolate=True, axis=1)
        #self._m = np.empty(self.lookup.shape)
        #for i, row in enumerate(self.lookup):
            #self._m[i] = pychip.pchip_slopes(x, row, kind='secant', tension=1)

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
