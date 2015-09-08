#!/home/jlaura/anaconda/bin/python

import glob
import logging
import os
import re
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI
import pvl

from app.date import astrodate, julian2ls, julian2season, date_convert
from app.fileio import io_gdal, io_json, io_utils
from app.wrappers import isiswrappers
from app.wrappers import pipelinewrapper
#from app.pvl import pvlparser
from app.utils import utils
from app.log import log
from app.interpolation import interpolator as interp
from app import config


#Constants
instrumentmap = {'THERMAL EMISSION IMAGING SYSTEM':'THEMIS'}  #Mapping of instrument names as stored in the header to short names
processingpipelines = {'themis_davinci':pipelinewrapper.themis_davinci}

def getincidence(parsestring):
    """
    Extract the incidence angle from a campt string
    """
    incidence_search = re.compile(r'\s*Incidence', flags=re.I)
    for l in parsestring.splitlines():
        if incidence_search.match(l):
            return float(l.split('=')[1].rstrip())

def getlatitude(parsestring):
    """
    Extract the planetocentric latitude angle from a campt string
    """
    latitude_search = re.compile(r'\s*PlanetocentricLatitude', flags=re.I)
    for l in parsestring.splitlines():
        if latitude_search.match(l):
            return float(l.split('=')[1].rstrip())

def radiance_to_temperature(image, jobs, workingpath):
    """
    Run the images through a processing pipeline to convert to temperature.

    Parameters
    ----------
    image       str PATH to the image to be processed
    jobs        dict Parameters used in the processing

    """
    pipeline = jobs['processing_pipeline']

    if pipeline == 'themis_davinci':
        if 'deplaid' in jobs.keys():
            #User defined deplaid
            deplaid = jobs['deplaid']
        else:
            #Fallback to standard deplaid
            deplaid = 1
            if deplaid == 'day':
                deplaid = 0
        return pipelinewrapper.themis_davinci(image,
                                                 int(jobs['uddw']),
                                                 int(jobs['tesatm']),
                                                 deplaid,
                                                 int(jobs['rtilt']),
                                                 int(jobs['force']),
                                                 workingpath)

def processimages(jobs, i):
    """
    Process a json object containing URIs for one or more jobs

    Parameters
    ----------
    jobs    dict    A dictionary of jobs containing an image list and
                    processing parameters

    """
    t1 = time.time()
    #Check the ISIS version
    isiswrappers.checkisisversion()

    image_path = jobs['images']
    if len(image_path) == 1:
        #Check if this is a directory
        if os.path.isdir(image_path[0]):
            image_path = glob.glob(os.path.join(image_path[0], '*'))

    #Pregenerate ancillary data structure
    ancillarydata = jobs['ancillarydata']
    for k, v in ancillarydata.iteritems():
        if v == 'None':
            logger.error('A {} dataset was not provided'.format(k))
            continue
        ancillarydata[k] = v  # readgeodata.GeoDataSet(v)

    #TODO Check with Trent - is this dumb to offer reprojection?
    #TODO Add reprojection attempt to get ancillary data into the right form.

    #Parse all of the input files and apply the parameters to each
    logger.info('Processing {} images.'.format(len(image_path)))
    #Ensure that the file exists at the PATH specified
    if os.path.isfile(jobs['images'][i]) == False:
        logging.error("Unable to find file: {}\n".format(jobs['images'][i]))
        sys.exit()

    logger.info('Reading image {}'.format(jobs['images'][i]))
    header = pvl.load(jobs['images'][i])
    bands = utils.find_in_dict(header, 'BAND_BIN_BAND_NUMBER')
    #bands = header['BAND_BIN_BAND_NUMBER']

    #Extract the instrument name
    if not 'name' in jobs.keys() or jobs['name'] == None:
        instrument = utils.find_in_dict(header, 'INSTRUMENT_NAME')
        jobs['name'] = instrumentmap[instrument]

    #Check that the required bands are present
    if not utils.checkbandnumbers(bands, jobs['bands']):
        logger.error("Image {} contains bands {}.  Band(s) {} must be present.\n".format(i, bands, jobs['bands']))
        sys.exit()

    if 'kerneluri' in jobs['projection'].keys():
        kernel = jobs['projection']['kerneluri']
    else:
        kernel = None
    #Checks passed, create a temporary working directory
    workingpath = io_utils.create_dir(basedir=jobs['workingdir'])
    #Convert to ISIS
    isiscube = isiswrappers.thm2isis(jobs['images'][i], workingpath)

    #TODO I am running spiceinit and campt only to get the incidence angle - what about running a database

    #Spiceinit
    isiswrappers.spiceinit(isiscube,kernel)

    #Campt - To get incidence angle and planetocentric clat
    campt = isiswrappers.campt(isiscube)
    incidence = getincidence(campt)
    clat = getlatitude(campt)
    logger.info("The planetocentric CLAT for the image is: {}".format(clat))

    deplaid = utils.checkdeplaid(incidence)
    logger.info("If deplaid is set in the input parameters, using {} deplaid routines".format(deplaid))

    #Process temperature data using some pipeline
    isiscube = radiance_to_temperature(jobs['images'][i], jobs, workingpath)

    #Processing the temperature to a level2 image
    isiswrappers.spiceinit(isiscube, kernel)
    isiscube = isiswrappers.cam2map(isiscube)
    header = pvl.load(isiscube)

    ulx = maxlat = utils.find_in_dict(header, 'MaximumLatitude')
    uly = utils.find_in_dict(header, 'MinimumLongitude')
    lrx = minlat = utils.find_in_dict(header, 'MinimumLatitude')
    lry = utils.find_in_dict(header, 'MaximumLongitude')

    logger.info('Processing ISIS cube: {}.'.format(isiscube))
    #Get the temperature array
    temperature = io_gdal.GeoDataSet(isiscube)
    processing_resolution = temperature.xpixelsize
    temparray = temperature.readarray()
    tempshape = temparray.shape
    logger.info('Themis temperature data has {} lines and {} samples'.format(tempshape[0], tempshape[1]))
    srs = temperature.spatialreference.ExportToWkt()
    gt = temperature.geotransform
    logger.info('The input temperature image projection is: {}'.format(srs))

    #Compute the extent and the resolution
    extent = subprocess.check_output(['/bin/bash', '-i', '-c', 'gdal_extent {}'.format(isiscube)])
    extent = extent.split()
    ext = [extent[0], extent[3], extent[2], extent[1]]
    logger.debug('GDAL computed extent as {} (xmin, ymin, xmax, ymax).'.format(ext))
    res = subprocess.check_output(['/bin/bash', '-i', '-c', 'gdal_resolution {}'.format(isiscube)])
    logger.debug('GDAL computed resolution is: {}'.format(res))

    #Iterate through the ancillary data.  Clip and resample to the input image
    for k, v in ancillarydata.iteritems():
            if isinstance(v, int) or isinstance(v, float):
                #The user wants a constant value to be used
                arr = np.empty(tempshape, dtype=np.float32)
                arr[:] = v
                ancillarydata[k] = arr
                logger.debug('{} set to a constant value, {}'.format(k, v))
                del arr
            elif v != 'None':
                basename = os.path.basename(v)
                root, extension = os.path.splitext(basename)
                #Clip and resample the image to the correct resolution
                tif = os.path.join(workingpath, root + '.tif')
                cmd = ['/usgs/apps/anaconda/bin/gdalwarp', '-overwrite',
                       '-r', 'bilinear', '-tr'] + res.split()\
                    + ['-te'] + ext + ['-t_srs'] + [srs]\
                    + [ '-of',  'GTiff', v, tif]
                warp_response = subprocess.check_output(cmd)

                #Read the resampled tif and extract the array
                ancillarydata[k] = io_gdal.GeoDataSet(tif)
                logger.debug('Dataset {} extract.'.format(v))

    bands = utils.find_in_dict(header,'BandNumber')
    logger.debug('Input TI image has bands {}'.format(bands))
    logger.debug('Input TI image LAT range is {} to {}'.format(minlat, maxlat))

    #TODO: This would be nicely abstracted to a transformation chain, a la VisPy 3d transforms
    #Time Parsing
    starttime = utils.find_in_dict(header, 'StartTime')
    stoptime = utils.find_in_dict(header, 'StopTime')

    #Convert UTC to Julian
    starttime_julian = astrodate.utc2jd(starttime)
    stoptime_julian = astrodate.utc2jd(stoptime)
    logger.debug('Input TI image time range is {} to {} (Julian)'.format(starttime_julian, stoptime_julian))

    #LsubS
    startlsubs, startmartianyear = julian2ls.julian2ls(starttime_julian)
    stoplsubs, stopmartianyear = julian2ls.julian2ls(stoptime_julian)
    season, startseason, stopseason = julian2season.j2season(startlsubs)

    logger.debug('Input TI image time range is {} / {} to {} / {} (LsubS)'.format(startlsubs[0],startmartianyear[0],
                                            stoplsubs[0], stopmartianyear[0]))
    season = season[0]
    #Pack the initial parameters for shipping to the interpolator
    parameters = {'starttime':starttime,
                  'stoptime':stoptime,
                  'startseason': startseason,
                  'stopseason':stopseason,
                  'season':season,
                  'startlatitude':minlat,
                  'stoplatitude':maxlat}
    t2 = time.time()
    logger.info('Data pre=processing, clipping, and map projection took {} seconds.'.format(t2 - t1))
    return temperature, parameters, ancillarydata, workingpath

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if rank == 0:
        #Setup logging
        loglevel = config.LOG_LEVEL
        log.setup_logging(default_path='../logging.yaml',
                          default_level=loglevel)

        logger = logging.getLogger(__name__)

        #Parse the job input
        if len(sys.argv) < 2:
            logger.error("Please supply an input configuration file.")
            sys.exit()
        logger.info("Processing using {} cores".format(comm.size))
        jobs = io_json.read_json(sys.argv[1])
        njobs = len(jobs['images'])
    else:
        njobs = None
    njobs = comm.bcast(njobs, root=0)
    for i in range(njobs):
        if rank == 0:
            temperature, parameters, ancillarydata, workingpath = processimages(jobs, i)
            t1 = time.time()
            ephemeris_data = interp.EphemerisInterpolator(temperature, ancillarydata, parameters)
            lookup = ephemeris_data.data
            logger.debug('Lookup table dtype is: {}, with a shape of: {}'.format(lookup.dtype, lookup.shape))
            lookup_shape = lookup.shape
            latitude_nodes = ephemeris_data.latitudenodes
            t2 = time.time()
            logger.info('Extracting lookup table and linear season interpolation took {} seconds'.format(t2 - t1))

            #Extract all of the necessary input data arrays
            td_g = temperature.readarray()
            y, x = td_g.shape
            result_cube = np.empty((y, x, 8), dtype=np.float32)
            result_cube[:,:,2] = td_g

            #TODO: Magic number here for m to km conversion is poor form.
            if not isinstance(ancillarydata['elevation'], np.ndarray):
                ed_g = result_cube[:,:,3] = ancillarydata['elevation'].readarray() / 1000.0
            else:
                ed_g = result_cube[:,:,4] = ancillarydata['elevation']

            if not isinstance(ancillarydata['slope'], np.ndarray):
                sd_g = result_cube[:,:,6] = ancillarydata['slope'].readarray()
            else:
                sd_g = result_cube[:,:,6] = ancillarydata['slope']

            if not isinstance(ancillarydata['slopeazimuth'], np.ndarray):
                sz_g = result_cube[:,:,5] = ancillarydata['slopeazimuth'].readarray()
            else:
                sz_g = result_cube[:,:,5] = ancillarydata['slopeazimuth']

            if not isinstance(ancillarydata['albedo'], np.ndarray):
                ad_g = result_cube[:,:,6] = ancillarydata['albedo'].readarray()
            else:
                ad_g = result_cube[:,:,7] = ancillarydata['albedo']

            if not isinstance(ancillarydata['dustopacity'], np.ndarray):
                od_g = result_cube[:,:,7] = ancillarydata['dustopacity'].readarray()
            else:
                od_g = result_cube[:,:,7] = ancillarydata['dustopacity']

            quotient, remainder = divmod(y, comm.size)
            qrs = (quotient, remainder, td_g.shape)
            ndv = temperature.ndv
            logger.debug('The input temperature dataset has a NoDataValue of {}'.format(ndv))
            temp_fname = temperature.filename
            logger.debug('Generated all data structures for communication.')
        else:
            #Allocate memory in child processes
            lookup_shape = None
            latitude_nodes = None
            td_g = ed_g = sd_g = sz_g = ad_g = od_g = None
            qrs = None
            ndv = None
            temp_fname = None

        #Broadcast scalars
        lookup_shape = comm.bcast(lookup_shape, root=0)
        latitude_nodes = comm.bcast(latitude_nodes, root=0)
        quotient, remainder, shape = comm.bcast(qrs, root=0)
        ndv = comm.bcast(ndv, root=0)
        temp_fname = comm.bcast(temp_fname, root=0)
        comm.Barrier()

        if rank == 0:
            logger.debug('Transmitted scalars and temperature geoobject to all cores.')
            tb1 = time.time()

        #Broadcast the ephemeris data (lookup tables)
        if rank != 0:
            lookup = np.empty((13500, 7), dtype=np.float64)
            temperature = io_gdal.GeoDataSet(temp_fname)
        comm.Bcast( [lookup, MPI.DOUBLE])

        if rank == 0:
            tb2 = time.time()
            logger.debug('Broadcast ephemeris data to all cores in {} seconds.'.format(tb2-tb1))

        #Compute the scatter offsets and scatter the input datasets
        if rank == 0:
            localshape = (quotient + remainder, shape[1])
        else:
            localshape = (quotient, shape[1])

        td = np.empty(localshape, dtype=np.float32)
        ed = np.empty(localshape, dtype=np.float32)
        sd = np.empty(localshape, dtype=np.float32)
        sz = np.empty(localshape, dtype=np.float32)
        ad = np.empty(localshape, dtype=np.float32)
        od = np.empty(localshape, dtype=np.float32)

        #Compute the scatter and gather offsets
        scattersize = list([(quotient + remainder) * shape[1]]) +\
                      [quotient * shape[1] for i in range(comm.size-1)]
        scatteroffsets = [0] + (np.cumsum(scattersize)[:-1]).tolist()

        for g, l in [(td_g, td), (ed_g, ed), (sd_g, sd), (sz_g, sz),
                     (ad_g, ad), (od_g, od)]:
            comm.Scatterv([g, scattersize, scatteroffsets, MPI.FLOAT],
                          [l, MPI.FLOAT])

        comm.Barrier()
        if rank == 0:
            tb3 = time.time()
            logger.debug('Scatter input data to all cores in {} seconds.'.format(tb3 - tb2))

        #Compute the start pixel for each chunk
        startpixel = rank * quotient
        if rank == 0:
            startpixel += remainder
            ta = time.time()

        param_interp = interp.ParameterInterpolator(temperature, td, ed, sd,
                                                    sz, od, ad, lookup,
                                                    latitude_nodes, startpixel)
        #Begin the interpolation
        param_interp.bruteforce()

        if rank != 0:
            result = None
        if rank == 0:
            result = np.empty(td_g.shape, dtype = np.float32)
        comm.Gatherv([param_interp.resultdata, MPI.FLOAT],
                     [result, scattersize, scatteroffsets, MPI.FLOAT],
                     root=0)

        if rank == 0:
            result_cube[:,:,0] = result

            outpath = os.path.join(jobs['outpath'], temperature.basename)
            logger.debug('Writing to {}.tif'.format(outpath))
            io_gdal.array_to_raster(result, '{}.tif'.format(outpath),
                                         projection=temperature.spatialreference,
                                         geotransform=temperature.geotransform,
                                         ndv=temperature.ndv)

            tb = time.time()
            logger.debug("Parameter interpolation took {} seconds".format(tb-ta))

            #Cleanup
            logger.debug("Removing the remporary working directory at {}".format(workingpath))
            io_utils.delete_dir(workingpath)
            logger.info("Processing image {} required {} seconds".format(temperature.basename, tb - ta))
