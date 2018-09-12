from distutils import dir_util
import json
import os
import unittest
from unittest.mock import Mock
from .. import interpolator
import datetime

from plio.io.io_gdal import GeoDataset
from plio.io.io_hdf import HDFDataset

import datetime
import numpy as np

from scipy.interpolate import interp1d, PchipInterpolator

#Seed the RNG
np.random.seed(12345)

import pytest

def generate_ancillary_data():
    arrayshape = (100,100)
    ancillarydata = {}
    ancillarydata['slope'] = np.zeros(arrayshape)
    ancillarydata['dustopacity'] = np.empty(arrayshape).fill(0.29457903)
    ancillarydata['slopeazimuth'] = np.zeros(arrayshape)
    ancillarydata['emissivity'] = np.ones(arrayshape)

    #Mock in the albedo dataset
    albedo = np.random.choice(np.arange(0.08, 0.32, 0.01), arrayshape)
    albedomock = Mock(spec=GeoDataset)
    albedomock.read_array.return_value = albedo
    ancillarydata['albedo'] = albedomock

    #Mock in the elevation dataset
    elevation = np.random.choice(np.arange(-1, 1, 0.1), arrayshape)
    elevationmock = Mock(spec=GeoDataset)
    elevationmock.read_array.return_value = elevation
    ancillarydata['elevation'] = elevationmock

    return ancillarydata

@pytest.fixture
def datadir(tmpdir, request):
    '''
    Fixture responsible for searching a folder with the same name of test
    module and, if available, moving all contents to a temporary directory so
    tests can use them freely.
    '''
    filename = request.module.__file__
    test_dir, _ = os.path.splitext(filename)
    if os.path.isdir(test_dir):
        dir_util.copy_tree(test_dir, str(tmpdir))
    return tmpdir

def ephemeris_parameters(datadir):
    parameters = []
    with open(datadir.join('expected_ephemeris.json').strpath) as fix:
        parameters_to_test = json.load(fix)

    for k, v in parameters_to_test.items():
        names = []
        args =[]
        for a, b in v.items():
            names.append(a)
            args.append(b)
        parameters.append((", ".join(names), args))
        #parameters.append((",".join(v.keys()), v.values()))
    return parameters

@pytest.fixture
def eph():
    temperature = Mock(spec=GeoDataset)
    ancillarydata = generate_ancillary_data()
    parameters = {'season': 10.072572590724686,
                  'startseason': 10,
                  'stopseason': 11,
                  'stoptime': 23.78,
                  'starttime': 23.65,
                  'startlatitude': -13.539651921201,
                  'stoplatitude': -7.4386175386755}
    hdffile = {'season_10':np.random.random((20,20)),
               'season_11':np.random.random((20,20))}

    return interpolator.EphemerisInterpolator(temperature,
                                              ancillarydata,
                                              parameters,
                                              hdffile)


def test_get_constant_emissivity_offsets(eph):
    start, stop = eph.get_emissivity_offsets()
    assert start == np.s_[27000:]
    assert stop == np.s_[27000:]

@pytest.mark.parametrize("starttime, stoptime, truth", [((2000,1,2,4,48,0),
                                                         (2000,1,2,4,56,0),
                                                         np.array([4, 5, 6, 7, 8])),
                                                        ((2000,1,2,23,56,0),
                                                         (2000,1,3,0,1,0),
                                                         np.array([16,17, 0, 1, 2])),
                                                        ((2000,1,2,23,30,0),
                                                         (2000,1,3,23,56,0),
                                                         np.array([15,16, 17, 0, 1]))])
def test_get_hour_offsets(eph, starttime, stoptime, truth):
    """
    Test a time offset from the middle of the day.
    """
    eph.parameters['starttime'] =  datetime.datetime(*starttime)
    eph.parameters['stoptime'] = datetime.datetime(*stoptime)

    hourslice, st = eph.get_hour_offsets()
    np.testing.assert_array_equal(hourslice, truth)

@pytest.mark.parametrize("hourslice, hour, data, truth", [(np.array([-2, -1, 0, 1, 2]),
                                                           0,
                                                           np.arange(350).reshape(10, 5, 7),
                                                           np.arange(17, 367, 35)),
                                                          (np.array([-2, -1, 0, 1, 2]),
                                                           -0.35,
                                                           np.arange(350).reshape(10, 5, 7),
                                                           np.arange(11.832727, 361.832727, 35))])
def test_interpolate_hour(eph, hourslice, hour, data, truth):

    res = eph.interpolatehour(hourslice, hour, data)
    np.testing.assert_array_almost_equal(res[:,3], truth)

@pytest.mark.parametrize("startlat, stoplat, truth", [(-13.539651921201,
                                                      -7.4386175386755,
                                                      np.array([13, 14, 15, 16, 17, 18, 19]))])
def test_get_lat_offsets(eph, startlat, stoplat, truth):
    eph.parameters['startlatitude'] = startlat
    eph.parameters['stoplatitude'] = stoplat
    latslice = eph.get_lat_offsets()
    np.testing.assert_array_almost_equal(latslice, truth)


@pytest.mark.parametrize("hourslice", [(np.array([14,15,16,17,0])),
                                       (np.array([1,2,3,4,5])),
                                       (np.array([16,17,0,1,2]))])
def test_extract_season(eph, hourslice):
    season = 10
    startemiss = np.s_[:13500]
    latslice = np.array([16, 17, 18, 19, 20])
    eph.lookuptable['season_10'] = np.arange(26973000).reshape((40500, 18, 37))

    data = eph.extract_season(season, startemiss, hourslice, latslice)

    assert data.shape == (13500, 5, 5)

def test_interpolate_seasons(eph):

    eph.startdata = np.empty((4,2,3))
    eph.stopdata = np.empty((4,2,3))
    for i in range(4):
        eph.startdata[i].fill(i)
        eph.stopdata[i].fill(i + 0.5)
    eph.interpolateseasons()
    truth = np.empty((4,2,3))
    v = [ 0.0362863, 1.0362863, 2.0362863, 3.0362863]
    for i in range(4):
        truth[i].fill(v[i])
    np.testing.assert_array_almost_equal(eph.data, truth)



@pytest.mark.parametrize("starttime, stoptime, startlat, stoplat, res", [((2000,1,2,0,0,0),
                                                                          (2000,1,2,0,0,0),
                                                                          0,0,17982016), # All on node
                                                                         ((2000,1,1,23,45,0),
                                                                          (2000,1,2,0,1,0),
                                                                          0,0, 17982330.5), # Crossing midnight, off node
                                                                         ((2000,1,2,4,15),
                                                                          (2000,1,3,4,30,0),
                                                                          0,0, 17982182.5), # Non-crossing, off node
                                                                          ])
def test_interpolate_all_ephemeris(eph, starttime, stoptime, startlat, stoplat, res):
    eph.parameters['starttime'] = datetime.datetime(*starttime)
    eph.parameters['stoptime'] = datetime.datetime(*stoptime)
    eph.parameters['startlatitude'] = startlat
    eph.parameters['stoplatitude'] = stoplat

    eph.lookuptable = {'season_10':np.arange(26973000).reshape(40500, 18, 37),
                       'season_11':np.arange(26973000).reshape(40500, 18, 37)}
    eph.interpolate_ephemeris()
    data = eph.data
    assert data[0][0] == res


@pytest.fixture
def par_random():
    arrayshape = (100,100)
    slope = np.zeros(arrayshape)
    dustopacity = np.empty(arrayshape).fill(0.29457903)
    slopeazimuth = np.zeros(arrayshape)
    emissivity = np.ones(arrayshape)
    albedo = np.random.choice(np.arange(0.08, 0.32, 0.01), arrayshape)
    elevation = np.random.choice(np.arange(-1, 1, 0.1), arrayshape)
    temperaturedata = np.random.choice(np.arange(0,10,0.5), arrayshape)

    lookuptable = np.empty((4,7), dtype=np.float32)
    latitudenodes = np.arange(-25,10,5)

    temperature = Mock(spec=GeoDataset)
    temperature.ndv = 0.0
    startpixel = 0
    ndv = 0
    r_ndv = 0
    reference_name = 'temperature'
    return interpolator.ParameterInterpolator(temperature,
                                         temperaturedata, elevation,
                                         slope, slopeazimuth,
                                         dustopacity, albedo,
                                         lookuptable, latitudenodes,
                                         startpixel, ndv, r_ndv, reference_name)
@pytest.fixture
def par_onnode():
    array_shape = (100,100)
    slope = np.zeros(arrayshape)
    dustopacity = np.empty(arrayshape).fill(0.29457903)
    slopeazimuth = np.zeros(arrayshape)
    emissivity = np.ones(arrayshape)
    albedo = np.empty(arrayshape).fill(0.08)
    elevation  = np.empty(arrayshape).fill(-1)
    temperaturedata = np.empty.fill(150)
    lookuptable = np.arange(67500).reshape(13500, 5)
    latitudenodes = np.array([-10, -5, 0, 5, 10])
    temperature = Mock(spec=GeoDataset)
    temperature.ndv = 0.0
    startpixel = 0
    ndv = 0
    r_ndv = 0
    reference_name = 'temperature'
    return interpolator.ParameterInterpolator(temperature,
                                         temperaturedata, elevation,
                                         slope, slopeazimuth,
                                         dustopacity, albedo,
                                         lookuptable, latitudenodes,
                                         startpixel, ndv, r_ndv, reference_name)


@pytest.fixture
def latnodes():
    return np.arange(-25,10,5)

## Below here would benefit from having parameterized tests

def test_interpolate_latitude(latnodes):
    lookup = np.arange(94500).reshape(-1, 7)
    lat_f = PchipInterpolator(latnodes, lookup, extrapolate=False, axis=1)
    latitude = -10
    res = lat_f(latitude).reshape(5,5,3,3,3,20)

    #Right on the node - we do not test that pychip is right,
    # just that our arrays are right.
    np.testing.assert_array_equal(res.ravel(), lookup[:,3])

    """
    Now check that the reshaping is working as expected

    The logic here is that the array is reshaped such that
    3 rows(one for each albedo) of 20 entries comprise each
    other parameter combination.  Therefore, the first
    entry should be the first 60 entries of the vector.
    """
    np.testing.assert_array_equal(res[0, 0, 0, 0, :, :],np.arange(3, 423, 7).reshape(3, 20))

def test_interpolate_elevation(par_random):
    par = par_random
    lookup = np.arange(13500).reshape(5,5,3,3,3,20)
    elevation_interp_func = par.compute_interpolation_function(np.array([-5., -2, -1, 6, 8]),
                                                               lookup, 'cubic')

    #Test on a node
    a = par.apply_interpolation_func(-1, elevation_interp_func)
    assert (5,3,3,3,20) == a.shape
    np.testing.assert_array_almost_equal(a[0,0,0,:,:],np.arange(5400,5460).reshape(3,-1))

    #Test off a node
    a = par.apply_interpolation_func(0, elevation_interp_func)
    assert (5,3,3,3,20) == a.shape
    truth = np.array([[ 7177.17440977,  7178.17440977,  7179.17440977,  7180.17440977,
            7181.17440977,  7182.17440977,  7183.17440977,  7184.17440977,
            7185.17440977,  7186.17440977,  7187.17440977,  7188.17440977,
            7189.17440977,  7190.17440977,  7191.17440977,  7192.17440977,
            7193.17440977,  7194.17440977,  7195.17440977,  7196.17440977],
        [ 7197.17440977,  7198.17440977,  7199.17440977,  7200.17440977,
            7201.17440977,  7202.17440977,  7203.17440977,  7204.17440977,
            7205.17440977,  7206.17440977,  7207.17440977,  7208.17440977,
            7209.17440977,  7210.17440977,  7211.17440977,  7212.17440977,
            7213.17440977,  7214.17440977,  7215.17440977,  7216.17440977],
        [ 7217.17440977,  7218.17440977,  7219.17440977,  7220.17440977,
            7221.17440977,  7222.17440977,  7223.17440977,  7224.17440977,
            7225.17440977,  7226.17440977,  7227.17440977,  7228.17440977,
            7229.17440977,  7230.17440977,  7231.17440977,  7232.17440977,
            7233.17440977,  7234.17440977,  7235.17440977,  7236.17440977]])

    np.testing.assert_array_almost_equal(a[0,0,0,:,:], truth)

def test_interpolate_slopeaz(par_random):
    par = par_random
    lookup = np.arange(2700).reshape(5,3,3,3,20)
    slaz_interp_func = par.compute_interpolation_function(np.array([0., 75, 210, 285, 360]),
                                                               lookup, 'cubic')

    #Test on a node
    a = par.apply_interpolation_func(75, slaz_interp_func)
    assert (3,3,3,20) == a.shape
    np.testing.assert_array_almost_equal(lookup[1,0,0,:,:], a[0,0,:,:])

    #Test off a node
    a = par.apply_interpolation_func(128.45, slaz_interp_func)
    assert (3,3,3,20) == a.shape
    truth = np.array([[ 744.76999764,  745.76999764,  746.76999764,  747.76999764,
            748.76999764,  749.76999764,  750.76999764,  751.76999764,
            752.76999764,  753.76999764,  754.76999764,  755.76999764,
            756.76999764,  757.76999764,  758.76999764,  759.76999764,
            760.76999764,  761.76999764,  762.76999764,  763.76999764],
        [ 764.76999764,  765.76999764,  766.76999764,  767.76999764,
            768.76999764,  769.76999764,  770.76999764,  771.76999764,
            772.76999764,  773.76999764,  774.76999764,  775.76999764,
            776.76999764,  777.76999764,  778.76999764,  779.76999764,
            780.76999764,  781.76999764,  782.76999764,  783.76999764],
        [ 784.76999764,  785.76999764,  786.76999764,  787.76999764,
            788.76999764,  789.76999764,  790.76999764,  791.76999764,
            792.76999764,  793.76999764,  794.76999764,  795.76999764,
            796.76999764,  797.76999764,  798.76999764,  799.76999764,
            800.76999764,  801.76999764,  802.76999764,  803.76999764]])
    np.testing.assert_array_almost_equal(a[0,0,:,:], truth)

def test_interpolate_slope(par_random):
    par = par_random
    lookup = np.arange(540).reshape(3,3,3,20)
    sl_interp_func = par.compute_interpolation_function(np.array([0., 30, 60]),
                                                               lookup, 'quadratic')

    #Test on a node
    a = par.apply_interpolation_func(60, sl_interp_func)
    assert (3,3,20) == a.shape
    np.testing.assert_array_almost_equal(lookup[2,0,:,:], a[0,:,:])

    #Test off a node
    a = par.apply_interpolation_func(18.667, sl_interp_func)
    assert (3,3,20) == a.shape
    truth = np.array([[ 112.002,  113.002,  114.002,  115.002,  116.002,  117.002,
            118.002,  119.002,  120.002,  121.002,  122.002,  123.002,
            124.002,  125.002,  126.002,  127.002,  128.002,  129.002,
            130.002,  131.002],
        [ 132.002,  133.002,  134.002,  135.002,  136.002,  137.002,
            138.002,  139.002,  140.002,  141.002,  142.002,  143.002,
            144.002,  145.002,  146.002,  147.002,  148.002,  149.002,
            150.002,  151.002],
        [ 152.002,  153.002,  154.002,  155.002,  156.002,  157.002,
            158.002,  159.002,  160.002,  161.002,  162.002,  163.002,
            164.002,  165.002,  166.002,  167.002,  168.002,  169.002,
            170.002,  171.002]])
    np.testing.assert_array_almost_equal(a[0,:,:], truth)

def test_interpolate_tau(par_random):
    par = par_random
    lookup = np.arange(180).reshape(3,3,20)
    tau_interp_func = par.compute_interpolation_function(np.array([0.02, 0.3, 0.6]),
                                                               lookup, 'quadratic')

    #Test on a node
    a = par.apply_interpolation_func(.30, tau_interp_func)
    assert (3,20) == a.shape
    np.testing.assert_array_almost_equal(lookup[1,:,:], a[:,:])

    #Test off a node
    a = par.apply_interpolation_func(0.53371, tau_interp_func)
    assert (3,20) == a.shape
    truth = np.array([[ 107.12359202,108.12359202,109.12359202,110.12359202,111.12359202,
        112.12359202,113.12359202,114.12359202,115.12359202,116.12359202,
        117.12359202,118.12359202,119.12359202,120.12359202,121.12359202,
        122.12359202,123.12359202,124.12359202,125.12359202,126.12359202],
    [ 127.12359202,128.12359202,129.12359202,130.12359202,131.12359202,
        132.12359202,133.12359202,134.12359202,135.12359202,136.12359202,
        137.12359202,138.12359202,139.12359202,140.12359202,141.12359202,
        142.12359202,143.12359202,144.12359202,145.12359202,146.12359202],
    [ 147.12359202,148.12359202,149.12359202,150.12359202,151.12359202,
        152.12359202,153.12359202,154.12359202,155.12359202,156.12359202,
        157.12359202,158.12359202,159.12359202,160.12359202,161.12359202,
        162.12359202,163.12359202,164.12359202,165.12359202,166.12359202]])
    np.testing.assert_array_almost_equal(a, truth)

def test_interpolate_albedo(par_random):
    par = par_random
    lookup = np.arange(60).reshape(3,20)
    alb_interp_func = par.compute_interpolation_function(np.array([0.08, 0.22, 0.32]),
                                                               lookup, 'quadratic')

    #Test on a node
    a = par.apply_interpolation_func(0.08, alb_interp_func)
    assert (20,) == a.shape
    np.testing.assert_array_almost_equal(lookup[0,:], a)

    #Test off a node
    a = par.apply_interpolation_func(0.1921, alb_interp_func)
    assert (20,) == a.shape
    truth = np.array([ 15.26962143,  16.26962143,  17.26962143,  18.26962143,
            19.26962143,  20.26962143,  21.26962143,  22.26962143,
            23.26962143,  24.26962143,  25.26962143,  26.26962143,
            27.26962143,  28.26962143,  29.26962143,  30.26962143,
            31.26962143,  32.26962143,  33.26962143,  34.26962143])
    np.testing.assert_array_almost_equal(a, truth)

@pytest.mark.parametrize("temp, result, tau, alb, elev, lat, slaz", [(12, 39.9, 0.02, 0.08, -5, 0, 0), # On node
                                                                     (37, 142.2, 0.02, 0.08, -5, 0, 0), # On node
                                                                     (87, 1804.7, 0.02, 0.08, -5, 0, 0), # On node
                                                                     (2, 24, 0.02, 0.08, -5, 0, 0), # Lowest node
                                                                     (1, 0, 0.02, 0.08, -5, 0, 0), # Below lowest
                                                                     (97, 3000, 0.02, 0.08, -5, 0, 0), # Highest node
                                                                     (99, 3000, 0.02, 0.08, -5, 0, 0), # Above highest
                                                                     (14.5, 45.28, 0.02, 0.08, -5, 0, 0), # Between albedo nodes
                                                                     (17, 39.9, 0.02, 0.08, -5, 0, 0)]) #
def test_bruteforce(temp, result, tau, alb, elev, lat, slaz):
    arrayshape = (3,3)
    slope = np.zeros(arrayshape)
    dustopacity = np.empty(arrayshape)
    dustopacity.fill(tau)
    slopeazimuth = np.empty(arrayshape)
    slopeazimuth.fill(slaz)
    emissivity = np.ones(arrayshape)
    albedo = np.empty(arrayshape)
    albedo.fill(alb)
    elevation  = np.empty(arrayshape)
    elevation.fill(elev)
    temperaturedata = np.empty(arrayshape)
    temperaturedata.fill(temp)
    lookuptable = np.arange(67500).reshape(13500, 5)
    latitudenodes = np.array([-10, -5, 0, 5, 10])
    temperature = Mock(spec=GeoDataset)
    temperature.pixel_to_latlon.return_value = (lat,None)
    temperature.ndv = 0.0
    startpixel = 0
    ndv = 0
    r_ndv = 0
    reference_name = 'td'
    par = interpolator.ParameterInterpolator(temperature,temperaturedata, elevation, slope, slopeazimuth,
                                             dustopacity, albedo,lookuptable, latitudenodes,startpixel, ndv, r_ndv, reference_name)
    par.bruteforce()
    print(par.resultdata, np.log(par.resultdata))
    assert pytest.approx(par.resultdata[0][0], 0.001) == result
