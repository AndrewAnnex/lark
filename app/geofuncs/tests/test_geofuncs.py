import unittest
from osgeo import osr

from .. import converters


class TestPixelToLatLon(unittest.TestCase):
    def runTest(self):
	gt = (-65003.091409083, 99.090078367505, 0.0,
	      419051.94141618, 0.0, -99.090078367505)
	srs = osr.SpatialReference()
	wktsrs = """PROJCS["SimpleCylindrical MARS",
		    GEOGCS["GCS_MARS",
		    DATUM["D_MARS",
		    SPHEROID["MARS",3396190,0]],
		    PRIMEM["Reference_Meridian",0],
		    UNIT["degree",0.0174532925199433]],
		    PROJECTION["Equirectangular"],
		    	PARAMETER["latitude_of_origin",0],
			PARAMETER["central_meridian",-0.12545906006398],
			PARAMETER["standard_parallel_1",0],
			PARAMETER["false_easting",0],
			PARAMETER["false_northing",0]]"""
	srs.ImportFromWkt(wktsrs)
	lat, lon = converters.pixel_to_latlon(gt, srs, [124,256])
	self.assertAlmostEqual(lat, 6.64170207194, places=4, msg='Error in latitude assertion')
	self.assertAlmostEqual(lon, -1.01480854465, places=4, msg='Error in longitude assertion')

if __name__ == '__main__':
    unittest.main()
