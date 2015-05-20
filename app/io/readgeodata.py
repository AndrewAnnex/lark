#!/home/jlaura/anaconda/bin/python

import numpy as np
from osgeo import gdal
from osgeo import osr

NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}

GDAL2NP_CONVERSION = {}

for k, v in NP2GDAL_CONVERSION.iteritems():
    GDAL2NP_CONVERSION[v] = k

GDAL2NP_CONVERSION[1] = 'int8'

class GeoDataSet():

    def __init__(self, filename):
	self.coordinate_transform = None
	self.inverse_coordinate_transform = None

        self.filename = filename
        self.opendata()

        #Populate data attributes
        self.get_geotransform()
        self.getdtype()
        self.getpixelsize()
        self.getorigin()
        self.getrotation()
        self.getprojection()
        self.getndv()
	self.getstatistics()

    def getdtype(self):
        """
        Get the input data datatype
        """
        band = self.ds.GetRasterBand(1)
        dtype = band.DataType
        self.dtype = GDAL2NP_CONVERSION[dtype]

    def getndv(self):
        """
        Get the no data value for the image
        """
        band = self.ds.GetRasterBand(1)
        self.ndv = band.GetNoDataValue()

    def getorigin(self):
        """
        Get the origin of the input image
        """
        self.ulx = self.geotransform[0]
        self.uly = self.geotransform[3]

    def getprojection(self):
        self.projection = self.ds.GetProjection()
        self.srs = osr.SpatialReference()
        self.srs.ImportFromWkt(self.projection)

    def getpixelsize(self):
        """
        Get the pixel size of the input data
        """
        self.xpixelsize = self.geotransform[1]
        self.ypixelsize = self.geotransform[5]

    def reproject(self, targetproj, extent, targetsize):
        """
        Computes the extent of the raster in some target space.

        Used to reproject and resample a subset of this raster into some target
        space.

        Parameters
        -----------
        target      An OSR coordinate transformation object from source to target
        extent      A list of upper left and lower right coordinates
        """
        raise NotImplementedError

    def getrotation(self):
        """
        Get the affine transformation coefficients, e.g. rotation
        """
        self.xrotation = self.geotransform[2]
        self.yrotation = self.geotransform[4]

    def opendata(self):
        """
        Open a data file
        """
        self.ds = gdal.Open(self.filename)

    def get_geotransform(self):
        """
        Get the geotransformation for an image
        """
        self.geotransform = self.ds.GetGeoTransform()
        success, self.invgeotransform = gdal.InvGeoTransform(self.geotransform)

    def getstatistics(self, band=1):
	"""
	Get the minimum and maximum from band 1
 	"""
	bnd = self.ds.GetRasterBand(band)
	self.minimum = bnd.GetMinimum()
	self.maximum = bnd.GetMaximum()

    def latlon_to_pixel(self, *args):
        """
        Convert a geospatial coordinate to a pixel coordinate

        Paramaters
        ----------

        *args   list of tuples of x,y coordinates in the form (lat, long)
        """
	gt = self.geotransform
	if self.coordinate_transform == None:
	    srs_latlong = self.srs.CloneGeogCS()
            self.coordinate_transform = osr.CoordinateTransformation(srs_latlon, self.srs)
        pixels = []
        for a in args:
            lat = a[0]
            lon = a[1]
            px, py, pz = self.coordinate_transform.TransformPoint(lon, lat)

            x = int((px - gt[0]) / gt[1])
            y = int((py - gt[3]) / gt[5])

            pixels.append((x,y))

        return pixels

    def pixel_to_latlon(self, x, y):
	"""
	Convert from an x,y pixel pair into a lat/lon pair
	using the source spatial reference system and geotransformation
	parameters.

	Parameters
	----------
	x : int
	    The x pixel coordinate

	y : int
	    The y pixel coordinate

	Returns
	-------
	lat, lon : tuple
		   (lat, lon) coordinates for the piven pixel as measured
		   from the upper left corner.
	"""
	gt = self.geotransform
	if self.inverse_coordinate_transform == None:
	    srs_latlong = self.srs.CloneGeogCS()
	    self.inverse_coordinate_transform = osr.CoordinateTransformation(self.srs, srs_latlong)
	ulon = x * gt[1] + gt[0]
	ulat = y * gt[5] + gt[3]
	(lon, lat, _ ) = self.inverse_coordinate_transform.TransformPoint(ulon, ulat)
	return lat, lon

    def extractarray(self, pixels=None):
        """
        Extract the required data as a numpy array

	Parameters
	----------

	pixels	: list
		  [start, ystart, xstop, ystop]
        """
        band = self.ds.GetRasterBand(1)

        if pixels == None:
            self.array = band.ReadAsArray().astype(np.float32)
        else:
            xstart = pixels[0][0]
            ystart = pixels[0][1]
            xextent = pixels[1][0] - xstart
            yextent = pixels[1][1] - ystart
            self.array = band.ReadAsArray(xstart, ystart, xextent, yextent).astype(np.float32)
	self.shape = self.array.shape

    def resamplearray(self, resolution):
        """
        Super or subsample the data to match the desired resolution.
        """
        raise NotImplementedError


