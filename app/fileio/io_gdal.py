import os

import numpy as np
from osgeo import gdal
from osgeo import osr

import extract_metadata as em

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

class GeoDataSet(object):
    def __init__(self, filename):
        self.filename = filename
        self.ds = gdal.Open(filename)

    @property
    def basename(self):
        if not getattr(self, '_basename', None):
            self._basename = os.path.splitext(os.path.basename(self.filename))[0]
        return self._basename

    @property
    def geotransform(self):
        if not getattr(self, '_geotransform', None):
            self._geotransform = self.ds.GetGeoTransform()
        return self._geotransform

    @property
    def standardparallels(self):
        if not getattr(self, '_standardparallels', None):
            self._standardparallels = em.get_standard_parallels(self.spatialreference)
        return self._standardparallels

    @property
    def unittype(self):
        if not getattr(self, '_unittype', None):
            self._unittype = self.ds.GetRasterBand(1).GetUnitType()
        return self._unittype

    @property
    def spatialreference(self):
        if not getattr(self, '_srs', None):
            self._srs = osr.SpatialReference()
            self._srs.ImportFromWkt(self.ds.GetProjection())
            try:
                self._srs.MorphToESRI()
                self._srs.MorphFromESRI()
            except: pass

            #Setup the GCS
            self._gcs = self._srs.CloneGeogCS()
        return self._srs

    @property
    def geospatial_coordinate_system(self):
        if not getattr(self, '_gcs', None):
            self._gcs = self.spatialreference.CloneGeogCS()
        return self._gcs

    @property
    def latlon_extent(self):
        if not getattr(self, '_latlonextent', None):
            ext = self.extent
            llat, llon = self.pixel_to_latlon(ext[0][0], ext[0][1])
            ulat, ulon = self.pixel_to_latlon(ext[1][0], ext[1][1])
            self._latlonextent = [(llat, llon), (ulat, ulon)]
        return self._latlonextent

    @property
    def extent(self):
        if not getattr(self, '_extent', None):
            gt = self.geotransform
            minx = gt[0]
            maxy = gt[3]

            maxx = minx + gt[1] * self.ds.RasterXSize
            miny = maxy + gt[5] * self.ds.RasterYSize

            self._extent = [(minx, miny), (maxx, maxy)]

        return self._extent

    @property
    def xpixelsize(self):
        """
        Get the pixel size of the input data
        """
        if not getattr(self, '_xpixelsize', None):
            self._xpixelsize = self.geotransform[1]
        return self._xpixelsize

    @property
    def ypixelsize(self):
        if not getattr(self, '_ypixelsize', None):
            self._ypixelsize = self.geotransform[5]
        return self._ypixelsize

    @property
    def xrotation(self):
        if not getattr(self, '_xrotation', None):
            self._xrotation = self.geotransform[2]
        return self._xrotation

    @property
    def yrotation(self):
        if not getattr(self, '_yrotation', None):
            self._yrotation = self.geotransform[4]
        return self._yrotation

    @property
    def coordinate_transformation(self):
        if not getattr(self, '_ct', None):
            self._ct = osr.CoordinateTransformation(self.spatialreference,
                                                  self.geospatial_coordinate_system)
        return self._ct

    @property
    def inverse_coordinate_transformation(self):
        if not getattr(self, '_ict', None):
                       self._ict = osr.CoordinateTransformation(self.geospatial_coordinate_system,
                                                                self.spatialreference)
        return self._ict

    @property
    def ndv(self, band=1):
        if not getattr(self, '_ndv', None):
            self._ndv = self.ds.GetRasterBand(band).GetNoDataValue()
        return self._ndv

    @property
    def scale(self):
        if not getattr(self, '_scale', None):
            unitname = self.spatialreference.GetLinearUnitsName()
            value = self.spatialreference.GetLinearUnits()
            self._scale = (unitname, value)
        return self._scale

    @property
    def spheroid(self):
        if not getattr(self, '_spheroid', None):
            self._spheroid = em.get_spheroid(self.spatialreference)
        return self._spheroid

    @property
    def rastersize(self):
        if not getattr(self, '_rastersize', None):
            self._rastersize = (self.ds.RasterXSize, self.ds.RasterYSize)
        return self._rastersize

    @property
    def central_meridian(self):
        if not getattr(self, '_central_meridian', None):
            self._central_meridian = em.get_central_meridian(self.spatialreference)
        return self._central_meridian

    def pixel_to_latlon(self, x, y):
        """
        Convert from pixel space to lat/lon space

        Parameters
        ----------
        x : float
            x-coordinate
        y : float
            y-coordinates

        Returns
        -------
        lat, lon : tuple
                   Latitude, Longitude
        """
        gt = self.geotransform
        x = gt[0] + (x * gt[1]) + (y * gt[2])
        y = gt[3] + (x * gt[4]) + (y * gt[5])
        lon, lat, _ = self.coordinate_transformation.TransformPoint(x, y)

        return lat, lon

    def latlon_to_pixel(self, lat, lon):
        gt = self.geotransform
        ulat, ulon, _ = self.inverse_coordinate_transformation.TransformPoint(lon, lat)
        x = (ulat - gt[0]) / gt[1]
        y = (ulon - gt[3]) / gt[5]
        return x, y

    def readarray(self, band=1, pixels=None, dtype='float32'):
        """
        Extract the required data as a numpy array

	Parameters
	----------

	pixels	: list
		  [start, ystart, xstop, ystop]

        dtype : str
                numpy dtype, e.g. float32
        """
        band = self.ds.GetRasterBand(band)

        dtype = getattr(np, dtype)

        if pixels == None:
            array = band.ReadAsArray().astype(dtype)
        else:
            xstart = pixels[0][0]
            ystart = pixels[0][1]
            xextent = pixels[1][0] - xstart
            yextent = pixels[1][1] - ystart
            array = band.ReadAsArray(xstart, ystart,
                                          xextent, yextent).astype(dtype)
        return array

def array_to_raster(array, filename, projection=None,
                    geotransform=None, outformat='GTiff',
                    ndv=None):
    driver = gdal.GetDriverByName(outformat)
    try:
        y, x, bands = array.shape
        single = False
    except:
        bands = 1
        y, x = array.shape
        single = True

    #This is a crappy hard code to 32bit.
    ds = driver.Create(filename, x, y, bands, gdal.GDT_Float64)

    if geotransform:
        ds.SetGeoTransform(geotransform)

    if projection:
        if isinstance(projection, str):
            ds.SetProjection(projection)
        else:
            ds.SetProjection(projection.ExportToWkt())

    if single == True:
        bnd = ds.GetRasterBand(1)
        if ndv != None:
            bnd.SetNoDataValue(ndv)
        bnd.WriteArray(array)
        ds.FlushCache()
    else:
        for i in range(1, bands + 1):
            bnd = ds.GetRasterBand(i)
            if ndv != None:
                bnd.SetNoDataValue(ndv)
            bnd.WriteArray(array[:,:,i - 1])
            ds.FlushCache()
