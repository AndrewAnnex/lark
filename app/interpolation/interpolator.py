from datetime import timedelta
import logging

import numpy as np

from app.io import readhdf, readgeodata
from app import config

logger = logging.getLogger('ThemisTI')

class Interpolator(object):

    emissivities = {0.85: (0, 13500),
                    0.90: (13500, 27000),
		    1.0: (27000, int(1e10))}

    elevations = {-5: (0, 2700),
                  -2: (2700, 5400),
                  -1: (5400, 8100),
                  6: (8100, 10800),
                  8: (13500, int(10e6))}

    time_lookup = {0:0, 3:1, 6:2, 7:3, 8:4, 9:5, 10:6,
                12:7, 14:8, 16:9, 24:10, 28:11, 32:12,
                35:13, 36:14, 40:15, 44:16, 47:17}

    latitude_lookup = dict(zip(range(-90, 95, 5), range(37)))

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

	"""

	self.temperature = temperature
        self.ancillarydata = ancillarydata
        self.parameters = parameters
	self.hdffile = readhdf.HDFDataSet()
	self.lookuptable = self.hdffile.data

        startseason = self.parameters['startseason']
        stopseason = self.parameters['stopseason']

	self.startlookup = self.lookuptable['season_{}'.format(startseason)]
	self.stoplookup = self.lookuptable['season_{}'.format(stopseason)]

	self.get_emissivity_offsets()
        self.get_hour_offsets()
        self.get_lat_offsets()

        #Double slicing is need because h5py does not support multiple fancy indices
        self.startdata = self.startlookup[:,
                                     self.latslice,
                                     self.startemiss[0]:self.startemiss[1]]

        self.startdata = self.startdata[self.hourslice, :, :]
        shape = self.startdata.shape
        nbytes = self.startdata.nbytes
        logger.debug('Seasonal arrays have shape {} and require {} bytes each'.format(shape, nbytes))

        self.stopdata = self.stoplookup[:,
                                    self.latslice,
                                    self.startemiss[0]:self.startemiss[1]]
        self.stopdata = self.stopdata[self.hourslice, :, :]

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
