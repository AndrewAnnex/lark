from osgeo import osr

def pixel_to_latlon(geotransform, srs, pixel):
    """
    Convert from pixel space to lat / lon space

    Parameters
    ----------
    geotransform : obj
		   a GDAL geotransformation object

    srs : obj
	 a GDAL spatiral reference system object

    pixel : iterable
	    (x,y) pixel space coordinates

    Returns
    -------

      : tuple
        (x,y) lat/lon coordinates

    ToDo
    ----
    Is it possible cache the srslatlon and the ct?
    """

    srslatlon = srs.CloneGeogCS()
    ct = osr.CoordinateTransformation(srs, srslatlon)
    ulon = pixel[0] * geotransform[1] + geotransform[0]
    ulat = pixel[1] * geotransform[5] + geotransform[3]
    (lon, lat, _) = ct.TransformPoint(ulon, ulat)
    return (lat, lon)
