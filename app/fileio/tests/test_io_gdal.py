import os
import unittest
import numpy as np
from osgeo import osr

from app.examples import get_path
from .. import io_gdal


class TestMercator(unittest.TestCase):
    def setUp(self):
        self.ds = io_gdal.GeoDataSet(get_path('Mars_MGS_MOLA_ClrShade_MAP2_0.0N0.0_MERC.tif'))

    def test_geotransform(self):
        self.assertEqual(self.ds.geotransform, (0.0, 4630.0, 0.0, 3921610.0, 0.0, -4630.0))

    def test_getunittype(self):
        #Write a test that has a unittype or check why this is not 'm'
        self.assertEqual(self.ds.unittype, '')

    def test_getextent(self):
        self.assertEqual(self.ds.extent, [(0.0, -3921610.0), (10667520.0, 3921610.0)])

    def test_getNDV(self):
        self.assertEqual(self.ds.ndv, 0.0)

    def test_pixel_to_latlon(self):
        lat, lon = self.ds.pixel_to_latlon(0,0)
        self.assertAlmostEqual(lat, 55.3322890518, 6)
        self.assertAlmostEqual(lon, 0.0, 6)

    def test_scale(self):
        self.assertEqual(self.ds.scale, ('Meter', 1.0))

    def test_extent(self):
        extent = self.ds.extent
        self.assertEqual(extent, [(0.0, -3921610.0), (10667520.0, 3921610.0)])

    def test_latlonextent(self):
        self.assertEqual(self.ds.latlon_extent, [(90.0, 0.0), (-90.0, -150.4067721290261)])

    def test_spheroid(self):
        sphere = self.ds.spheroid
        self.assertAlmostEqual(sphere[0], 3396190.0, 6)
        self.assertEqual(self.ds.spheroid, (3396190.0, 3376200.0, 169.8944472236118))
        self.assertAlmostEqual(sphere[1], 3376200.0, 6)
        self.assertAlmostEqual(sphere[2], 169.8944472236118, 6)

    def test_rastersize(self):
        size = self.ds.rastersize
        self.assertEqual(size[0], 2304)
        self.assertEqual(size[1], 1694)

    def test_basename(self):
        self.assertEqual(self.ds.basename, 'Mars_MGS_MOLA_ClrShade_MAP2_0.0N0.0_MERC')

    def test_xpixelsize(self):
        self.assertAlmostEqual(self.ds.xpixelsize, 4630.0, 6)
       
    def test_readarray(self):
        arr = self.ds.readarray()
        self.assertEqual(arr.shape, (1694, 2304))
        self.assertEqual(arr.dtype, np.float32)
    
    def test_read_clipped_array(self):
        arr = self.ds.readarray(pixels=((0,0), (100,100)))
        self.assertEqual(arr.shape, (100,100))

    def test_readarray_setdtype(self):
        arr = self.ds.readarray(dtype='int8')
        self.assertEqual(arr.dtype, np.int8)
        self.assertAlmostEqual(np.mean(arr), 10.10353227, 6)

class TestLambert(unittest.TestCase):
    def setUp(self):
        self.ds = io_gdal.GeoDataSet(get_path('Lunar_LRO_LOLA_Shade_MAP2_90.0N20.0_LAMB.tif'))

    def test_geotransform(self):
        self.assertEqual(self.ds.geotransform, (-464400.0, 3870.0, 0.0, -506970.0, 0.0, -3870.0))

    def test_getunittype(self):
        #Write a test that has a unittype or check why this is not 'm'
        self.assertEqual(self.ds.unittype, '')

    def test_getextent(self):
        self.assertEqual(self.ds.extent, [(-464400.0, -1571220.0), (460530.0, -506970.0)])

    def test_getNDV(self):
        self.assertEqual(self.ds.ndv, 0.0)

    def test_pixel_to_latlon(self):
        lat, lon = self.ds.pixel_to_latlon(0,0)
        self.assertAlmostEqual(lat, 69.90349154912009, 6)
        self.assertAlmostEqual(lon, -29.72166902463681, 6)

    def test_latlon_to_pixel(self):
        lat, lon = 69.90349154912009, -29.72166902463681
        pixel = self.ds.latlon_to_pixel(lat, lon)
        self.assertAlmostEqual(pixel[0], 0.0, 6)
        self.assertAlmostEqual(pixel[1], 0.0, 6)

    def test_standard_parallels(self):
        sp = self.ds.standardparallels
        self.assertEqual(sp, [73.0, 42.0])

    def test_extent(self):
        extent = self.ds.extent
        self.assertEqual(extent, [(-464400.0, -1571220.0), (460530.0, -506970.0)])

    def test_latlon_extent(self):
        self.assertEqual(self.ds.latlon_extent, [(-89.98516988892511, -171.35800063907413), (-89.95883789218114, -178.8099427811737)])

class TestPolar(unittest.TestCase):
    def setUp(self):
        self.ds = io_gdal.GeoDataSet(get_path('Mars_MGS_MOLA_ClrShade_MAP2_90.0N0.0_POLA.tif'))

    def test_geotransform(self):
        self.assertEqual(self.ds.geotransform, (-2129800.0, 4630.0, 0.0, 2129800.0, 0.0, -4630.0))

    def test_getunittype(self):
        #Write a test that has a unittype or check why this is not 'm'
        self.assertEqual(self.ds.unittype, '')

    def test_getextent(self):
        self.assertEqual(self.ds.extent, [(-2129800.0, -2129800.0), (2129800.0, 2129800.0)])

    def test_getNDV(self):
        self.assertEqual(self.ds.ndv, 0.0)

    def test_pixel_to_latlon(self):
        lat, lon = self.ds.pixel_to_latlon(0,0)
        self.assertAlmostEqual(lat, 42.2574735013, 6)
        self.assertAlmostEqual(lon, -135.0, 6)

    def test_latlon_to_pixel(self):
        lat, lon = 42.2574735013, -135.0
        pixel = self.ds.latlon_to_pixel(lat, lon)
        self.assertAlmostEqual(pixel[0], 0.0, 6)
        self.assertAlmostEqual(pixel[1], 0.0, 6)

    def test_extent(self):
        extent = self.ds.extent
        self.assertEqual(extent, [(-2129800.0, -2129800.0), (2129800.0, 2129800.0)])

class TestWriter(unittest.TestCase):
    def setUp(self):
        self.arr = np.random.random((100,100))
        self.ndarr = np.random.random((100,100,3))

    def test_write_arr(self):
        io_gdal.array_to_raster(self.arr, 'test.tif')
        self.assertTrue(os.path.exists('test.tif'))
        os.remove('test.tif')

    def test_write_ndarr(self):
        io_gdal.array_to_raster(self.arr, 'test.tif')
        self.assertTrue(os.path.exists('test.tif'))
        os.remove('test.tif')

    def test_with_geotrasform(self):
        gt =  (-464400.0, 3870.0, 0.0, -506970.0, 0.0, -3870.0)
        io_gdal.array_to_raster(self.arr, 'test.tif', geotransform = gt)
        ds = io_gdal.GeoDataSet('test.tif')
        self.assertEqual(gt, ds.geotransform)

    def test_with_ndv(self):
        ndv = 0.0
        #nd array
        io_gdal.array_to_raster(self.ndarr,'test.tif', ndv=ndv)
        ds = io_gdal.GeoDataSet('test.tif')
        self.assertEqual(ds.ndv, ndv)

        #array
        io_gdal.array_to_raster(self.arr,'test.tif', ndv=ndv)
        ds = io_gdal.GeoDataSet('test.tif')
        self.assertEqual(ds.ndv, ndv) 

    def test_with_projection(self):
        wktsrs = """PROJCS["Moon2000_Mercator180",
            GEOGCS["GCS_Moon_2000",
                DATUM["Moon_2000",
                    SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]],
                PRIMEM["Reference_Meridian",0.0],
                UNIT["Degree",0.017453292519943295]],
            PROJECTION["Mercator_1SP"],
            PARAMETER["False_Easting",0.0],
            PARAMETER["False_Northing",0.0],
            PARAMETER["Central_Meridian",180.0],
            PARAMETER["latitude_of_origin",0.0],
            UNIT["Meter",1.0]]"""
        io_gdal.array_to_raster(self.arr, 'test.tif', projection=wktsrs)
        expected_srs = """PROJCS["Moon2000_Mercator180",
            GEOGCS["GCS_Moon_2000",
                DATUM["Moon_2000",
                    SPHEROID["Moon_2000_IAU_IAG",1737400,0]],
                PRIMEM["Reference_Meridian",0],
                UNIT["Degree",0.017453292519943295]],
            PROJECTION["Mercator_1SP"],
            PARAMETER["central_meridian",180],
            PARAMETER["false_easting",0],
            PARAMETER["false_northing",0],
            UNIT["Meter",1],
            PARAMETER["latitude_of_origin",0.0]]"""
        ds = io_gdal.GeoDataSet('test.tif')
        test_srs = ds.spatialreference.__str__()
        self.assertEqual(test_srs.split(), expected_srs.split())

    def tearDown(self):
        try:
            os.remove('test.tif')
        except:
            pass



