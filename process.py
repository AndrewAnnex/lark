#!/home/jlaura/anaconda/bin/python

import datetime
import glob
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np

from app.date import astrodate, julian2ls
from app.io import fileio, readgeodata
from app.wrappers import isiswrappers
from app.wrappers import pipelinewrapper
from app.pvl import pvlparser

#Constants
DATEFMT = '%Y-%m-%dT%H:%M:%S.%f' #Date format from PVL matched to a Themis Cube
instrumentmap = {'THERMAL EMISSION IMAGING SYSTEM':'THEMIS'}  #Mapping of instrument names as stored in the header to short names
processingpipelines = {'themis_davinci':pipelinewrapper.themis_davinci}


def checkbandnumbers(bands, checkbands):
    """
    Given a list of input bands, check that the passed
    tuple contains those bands.

    In case of THEMIS, we check for band 9 as band 9 is the temperature
    band required to derive thermal inertia.  We also check for band 10
    which is required for TES atmosphere calculations.

    Parameters
    -----------
    bands       tuple of bands in the input image
    checkbands  list of bands to check against

    Returns
    --------
    Boolean     True if the bands are present, else False
    """
    for c in checkbands:
        if c not in bands:
            return False
    return True


def createlogger():
    """
    Create a Python logging object to log processing.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger('ThemisTI')
    logger.info('Starting to process Themis TI images')

    return logger

def checkdeplaid(incidence):
    """
    Given an incidence angle, select the appropriate deplaid method.

    Parameters
    -----------
    incidence       float incidence angle extracted from the campt results.

    """
    if incidence >= 95 and incidence <= 180:
        return 'night'
    elif incidence >=90 and incidence < 95:
        logger.error("Incidence angle is {}.  This is a twilight image, using night time deplaid.".format(incidence))
        return 'night'
    elif incidence >= 85 and incidence < 90:
        logger.error("Incidence angle is {}.  This is a twilight image, using daytime deplaid".format(incidence))
        return 'day'
    elif incidence >= 0 and incidence < 85:
        logger.error("Incidence angle is {}.  This is a daytime image, you may not want to use this product.".format(incidence))
    else:
        logger.error("Incidence does not fall between 0 and 180.")

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

def time_to_pydate(timestring):
    """
    Converts a PVL time string into a python datetime object

    Parameters
    ----------
    timestring	str A date / time string from a PVL header in DATEFMT form

    Returns
    --------
    time 	obj A python datetime object
    """

    return datetime.datetime.strptime(timestring, DATEFMT)


def createworkingdir(basedir='/scratch/jlaura/'):
    """
    Create a unique, temporary directory in /tmp where processing will occur

    Parameters
    ----------
    basedir     str The PATH to create the temporary directory in.
    """
    return tempfile.mkdtemp(dir=basedir)

def processimages(jobs):
    """
    Process a json object containing URIs for one or more jobs

    Parameters
    ----------
    jobs    dict    A dictionary of jobs containing an image list and
                    processing parameters

    """

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
        #projections.append(ancillarydata[k].projection)

    #if not all(i[0] for i in projections):
        #logger.error("Projections do not match on all ancillary data.  Please ensure all data is in Simple Cylindrical projection.")
        #sys.exit()
    #else:
        #logger.info("Ancillary data projections match, proceesing with processing.")

    #TODO Check with Trent - is this dumb to offer reprojection?
    #TODO Add reprojection attempt to get ancillary data into the right form.


    #Parse all of the input files and apply the parameters to each
    logger.info('Processing {} images.'.format(len(image_path)))
    for i in image_path:
        #Ensure that the file exists at the PATH specified
        if os.path.isfile(i) == False:
            logging.error("Unable to find file: {}\n".format(i))
            continue

        logger.info('Reading image {}'.format(i))
        header = pvlparser.getheader(i)
        bands = header['BAND_BIN_BAND_NUMBER']

        #Extract the instrument name
        if not 'name' in jobs.keys() or jobs['name'] == None:
            jobs['name'] = instrumentmap[header['INSTRUMENT_NAME']]

        #Check that the required bands are present
        if not checkbandnumbers(bands, jobs['bands']):
            logger.error("Image {} contains bands {}.  Band(s) {} must be present.\n".format(i, bands, jobs['bands']))
            continue

	if 'kerneluri' in jobs['projection'].keys():
            kernel = jobs['projection']['kerneluri']
	else:
	    kernel = None
        #Checks passed, create a temporary working directory
        workingpath = createworkingdir()

        #Convert to ISIS
        isiscube = isiswrappers.thm2isis(i, workingpath)

        #TODO I am running spiceinit and campt only to get the incidence angle - what about running a database

        #Spiceinit
        isiswrappers.spiceinit(isiscube,kernel)

        #Campt - To get incidence angle and planetocentric clat
        campt = isiswrappers.campt(isiscube)
        incidence = getincidence(campt)
        clat = getlatitude(campt)
        logger.info("The planetocentric CLAT for the image is: {}".format(clat))

        deplaid = checkdeplaid(incidence)
        logger.info("If deplaid is set in the input parameters, using {} deplaid routines".format(deplaid))

        #Process temperature data using some pipeline
        #TODO Fix davinci to processing works - errors on that side
        isiscube = radiance_to_temperature(i, jobs, workingpath)

        #Processing the temperature to a level2 image
        isiswrappers.spiceinit(isiscube, kernel)
        isiscube = isiswrappers.cam2map(isiscube)
        header = pvlparser.getheader(isiscube)

        ulx = maxlat = header['IsisCube']['Mapping']['MaximumLatitude']
        uly = header['IsisCube']['Mapping']['MinimumLongitude']
        lrx = minlat = header['IsisCube']['Mapping']['MinimumLatitude']
        lry = header['IsisCube']['Mapping']['MaximumLongitude']

        logger.info('Processing ISIS cube: {}.'.format(isiscube))

        #Get the temperature array
        temperature = readgeodata.GeoDataSet(isiscube)
        processing_resolution = temperature.xpixelsize
        temperature.extractarray()
        tempshape = temperature.array.shape
        logger.info('Themis temperature data has {} lines and {} samples'.format(tempshape[0], tempshape[1]))
        srs = temperature.srs.ExportToWkt()
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
                    arr = np.empty(tempshape, dtype=type(v))
                    arr[:] = v
                    ancillarydata[k] = arr
                    logger.debug('{} set to a constant value, {}'.format(k, v))
                    del arr
                elif v != 'None':
                    basename = os.path.basename(v)
                    root, extension = os.path.splitext(basename)
                    #Clip and resample the image to the correct resolution
                    tif = os.path.join(workingpath, root + '.tif')
                    cmd = ['/home/jlaura/anaconda/bin/gdalwarp', '-overwrite',
                           '-r', 'bilinear', '-tr'] + res.split()\
                        + ['-te'] + ext + ['-t_srs'] + [srs]\
                        + [ '-of',  'GTiff', v, tif]
                    warp_response = subprocess.check_output(cmd)
                    logger.debug(warp_response)

                    #Read the resampled tif and extract the array
                    ancillarydata[k] = readgeodata.GeoDataSet(tif)
                    ancillarydata[k].extractarray()
                    logger.debug('Dataset {} extract with shape {}'.format(v, ancillarydata[k].array.shape))
        bands = header['IsisCube']['BandBin']['BandNumber']
        logger.debug('Input TI image has bands {}'.format(bands))
        logger.debug('Input TI image LAT range is {} to {}'.format(minlat, maxlat))

        #Time Parsing
        starttime = time_to_pydate(header['IsisCube']['Instrument']['StartTime'])
        stoptime = time_to_pydate(header['IsisCube']['Instrument']['StopTime'])
        #Convert UTC to Julian
        starttime = astrodate.utc2jd(starttime)
        stoptime = astrodate.utc2jd(stoptime)
        logger.debug('Input TI image time range is {} to {} (Julian)'.format(starttime, stoptime))
        #LsubS
        startlsubs, startmartianyear = julian2ls.julian2ls(starttime)
        stoplsubs, stopmartianyear = julian2ls.julian2ls(stoptime)
        logger.debug('Input TI image time range is {} / {} to {} / {} (LsubS)'.format(startlsubs[0],startmartianyear[0],
                                                stoplsubs[0], stopmartianyear[0]))

	#Pack the initial parameters for shipping to the interpolator
        parameters = {'starttime':starttime,
		      'stoptime':stoptime,
		      'startlatitude':minlat,
		      'stoplatitude':maxlat}
	#Interpolation code is inserted here...
	return temperature, parameters, ancillarydata, workingpath



if __name__ == "__main__":
    logger = createlogger()
    if len(sys.argv) < 2:
	logger.error("Please supply an input configuration file.")
	sys.exit()
    jobs = fileio.readinputfile(sys.argv[1])
    temperature, parameters, ancillarydata, workingpath = processimages(jobs)
    print temperature
    print
    print parameters
    print
    print ancillarydata
    print
    print workingpath
    print
    print dir(temperature)

    print temperature.pixel_to_latlon(0,0)
    print temperature.pixel_to_latlon(temperature.shape[1] - 1, temperature.shape[0] - 1)

    shutil.rmtree(workingpath)
