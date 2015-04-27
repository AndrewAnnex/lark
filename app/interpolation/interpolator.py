from app.io import readhdf

class Interpolator(object):

    def __init__(self, temperature, ancillarydata, startseason, stopseason):

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
	self.hdffile = readhdf.HDFDataSet()
	self.lookuptable = self.hdffile.data

	self.startlookup = self.lookuptable['season_{}'.format(startseason)]
	self.stoplookup = self.lookuptable['season_{}'.format(stopseason)]

	print self.startlookup.shape

	"""
	Extract start and stop in only the required lat and lon boxes
	Make sure that an emissivity check occurs first since that can
	save 1/3 of the overall space.
	Check the size on the start and stop since they are 6GB / 80 each
	Interpolation should be doable using multiprocessing AND / OR MPI
	"""
