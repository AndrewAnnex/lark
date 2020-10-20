#!/usr/bin/env python

import glob
import logging
import os
import re
from shutil import copyfile
import subprocess
import sys
import time

import numpy as np
from mpi4py import MPI

from plio.io import io_gdal, io_hdf, io_json
from plio.date import astrodate, julian2ls, julian2season
import plio.utils
from plio.utils import log
from plio.utils.utils import check_file_exists, find_in_dict

import gdal

import pysis
import pvl

from krc.wrappers import pipelinewrapper, isiswrapper
from krc.utils import utils
from krc.interpolation import interpolator as interp
from krc import config

#Constants
instrumentmap = {'THERMAL EMISSION IMAGING SYSTEM':'THEMIS'}  #Mapping of instrument names as stored in the header to short names
processingpipelines = {'themis_davinci':pipelinewrapper.themis_davinci}

#Get MPI to abort cleanly in the case of an error
sys_excepthook = sys.excepthook
def mpi_excepthook(v, t, tb):
    sys_excepthook(v, t, tb)
    print(tb.stack)
    MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

#Setup logging
log.setup_logging(level=config.LOG_LEVEL)
logger = logging.getLogger(__name__)

def process_header(job):
    """
    Given the input image and job instructions, check that the necessary
    header information is present to process.

    Parameters
    ----------
    job : dict
          Containing the PATH to an images

    Returns
    -------
    job : dict
          With updated, image header specific values

    """

    header = pvl.load(job['images'])
    bands = find_in_dict(header, 'BAND_BIN_BAND_NUMBER')
    #bands = header['BAND_BIN_BAND_NUMBER']

    #Extract the instrument name
    if not 'name' in job.keys() or job['name'] == None:
        instrument = find_in_dict(header, 'INSTRUMENT_NAME')
        job['name'] = instrumentmap[instrument]

    #Check that the required bands are present
    if not utils.checkbandnumbers(bands, job['bands']):
        logger.error("Image {} contains bands {}.  Band(s) {} must be present.\n".format(i, bands, job['bands']))
        return

    if 'kerneluri' in job['projection'].keys():
        kernel = job['projection']['kerneluri']
    else:
        kernel = None

    return job

def processimage(job, workingpath, parameters):
    """
    Process a THEMIS EDR using ISIS and Davinci to a level 2 map projected
    product. putting the output and intermediary files into the workingpath.

    Parameters
    ----------
    job : dict
          A dictionary of job containing an image list and
          processing parameters

    workingpath : str
                  The working directory for intermediate files

    Returns
    -------

    isiscube : str
               PATH to the processed ISIS cube

    startlocaltime : str
                     The image start time

    stoplocaltime : str
                    The image stop time
    """
    t1 = time.time()

    # Check that the image exists
    image = job['images']
    if not check_file_exists(image):
        MPI.COMM_WORLD.Abort(1)

    #TODO Check with Trent - is this dumb to offer reprojection?
    #TODO Add reprojection attempt to get ancillary data into the right form.

    # Copy the file from the PDS nfs to scratch
    basepath, fname = os.path.split(image)
    tmp_local_image = os.path.join(workingpath, fname)
    copyfile(image, tmp_local_image)

    logger.info('Reading image {}'.format(tmp_local_image))

    # Process the image header
    job = process_header(job)
    if job is None:
        MPI.COMM_WORLD.Abort(1)

    fname, _ = os.path.splitext(tmp_local_image)
    #Convert to ISIS
    outcube = os.path.join(workingpath, '{}.cub'.format(fname))
    
    kernels = job.get('kernels', {})
    ckkernel = kernels.get('ckkernel', None)
    spkkernel = kernels.get('spkkernel', None)
    logger.debug('Using CK kernel: {}'.format(ckkernel))
    logger.debug('Using SPK kernel: {}'.format(spkkernel))
    isiswrapper.preprocess_for_davinci(image, outcube, ckkernel=ckkernel, spkkernel=spkkernel)
    res = isiswrapper.campt_header(outcube)
    incidence, stoplocaltime, startlocaltime = res

    # Process image Davinci
    deplaid = utils.checkdeplaid(incidence)
    logger.info("If deplaid is set in the input parameters, using {} deplaid routines".format(deplaid))

    if 'deplaid' in job.keys():
        #User defined deplaid, cast to int for Davinci
        deplaid = int(job['deplaid'])
    else:
        #Fallback to standard deplaid
        deplaid = 1
        if deplaid == 'day':
            deplaid = 0

    #Process temperature data using some pipeline
    #try:
    dvcube = processingpipelines[job['processing_pipeline']](image, workingpath, deplaid,
                                     job['uddw'], job['tesatm'], job['rtilt'], job['force'])
    #except:
    #    logger.error("Unknown processing pipeline: {}".format(job['processing_pipeline']))

    # Map project using isis
    isiscube = isiswrapper.postprocess_for_davinci(dvcube, ckkernel=ckkernel, spkkernel=spkkernel)

    parameters['startlocaltime'] = startlocaltime
    parameters['stoplocaltime'] = stoplocaltime

    return isiscube, parameters


def extract_metadata(isiscube, parameters):
    """
    Given a Davinci processed, level 2 cube, extract the necessary
    metadata from the header to support clipping supplemental data sets.
    """
    header = pvl.load(isiscube)

    ulx = maxlat = find_in_dict(header, 'MaximumLatitude')
    uly = find_in_dict(header, 'MinimumLongitude')
    lrx = minlat = find_in_dict(header, 'MinimumLatitude')
    lry = find_in_dict(header, 'MaximumLongitude')
    logger.debug('Input TI image LAT range is {} to {}'.format(minlat, maxlat))

    starttime = find_in_dict(header, 'StartTime')
    stoptime = find_in_dict(header, 'StopTime')

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
    logger.debug('Season: {}, Start Season: {}, Stop Season {}'.format(season, startseason, stopseason))

    parameters['startseason'] = startseason
    parameters['stopseason'] = stopseason
    parameters['season'] = season
    parameters['startlsubs'] = startlsubs
    parameters['starttime'] = starttime
    parameters['stoptime'] = stoptime
    parameters['startlatitude'] = lrx
    parameters['stoplatitude'] = ulx
    parameters['startmartianyear'] = startmartianyear
    parameters['stopmartianyear'] = stopmartianyear

    return parameters

def extract_temperature(isiscube, reference_dataset=None):
    """
    Extract the temperature data from the processed ISIS cube.

    Parameters
    ----------
    isiscube : str
               PATH to an ISIS cube to extract
    """
    temperature = io_gdal.GeoDataset(isiscube)
    processing_resolution = temperature.pixel_width
    tempshape = list(temperature.raster_size)[::-1]
    logger.info('Themis temperature data has {} lines and {} samples'.format(tempshape[0], tempshape[1]))
    srs = temperature.spatial_reference.ExportToWkt()
    logger.info('The input temperature image projection is: {}'.format(srs))
    return temperature

def extract_ancillary_data(job, temperature, parameters, workingpath, shape, reference_dataset):
    """
    For all ancillary data sets, extract the requested spatial extent

    Parameters
    ----------
    job : dict
          Job specification dictionary

    temperature : object
                  Plio GeoDataset object

    parameters : dict
                 of extent and time parameters
    """
    ancillarydata = job['ancillarydata']
    for k, v in ancillarydata.items():
        ancillarydata[k] = v

    # Flag to punt on the image if some ancillary data load fails.
    ancillary_failure = False

    #Iterate through the ancillary data.  Clip and resample to the input image
    for k, v in ancillarydata.items():
        print(k, v)
        dustopacity_band = 1
        if isinstance(v, int) or isinstance(v, float):
            #The user wants a constant value to be used
            arr = np.empty(shape, dtype=np.float32)
            arr[:] = v
            ancillarydata[k] = arr
            logger.debug('{} set to a constant value, {}'.format(k, v))
            del arr
            continue
        elif v == 'montabone':
                startls = parameters['startlsubs'][0]
                startmartianyear = int(parameters['startmartianyear'][0])
                files = glob.glob('/scratch/jlaura/KRC/basemaps/montabone/*.tif')

                if len(files) == 0:
                    logger.error('Unable to locate Montabone opacity maps.')
                    ancillary_failure = True
                    continue
                
                for f in files:
                    if str(startmartianyear) in f:
                        fname = f
                        break
                ds = gdal.Open(fname)
                time_steps = ds.GetMetadataItem('NETCDF_DIM_Time_VALUES')
                # Convert from a string to a list via a set due to how this metadata is stored
                time_steps = list(eval(time_steps))
                dustopacity_band = time_steps.index(min(time_steps, key=lambda x: abs(x-startls))) + 1
                v = fname
        """elif v == 'tes':
            startls = startlsubs[0]
            files = glob.glob('/scratch/jlaura/KRC/basemaps/tes_opac/*')
            ls = {}
            for t, f in enumerate(files):
                base, ext = os.path.splitext(f)
                ls[float(base.split('_')[-1][2:])] = t
            keys = []
            for key in ls.keys():
                try:
                    keys.append(float(key))
                except: pass

            key = min(keys, key=lambda x: abs(x-startls))
            v = files[ls[key]]"""
        tif = os.path.join(workingpath, os.path.basename(v))
        #Clip and resample the image to the correct resolution
        v = io_gdal.GeoDataset(v)
        print(v, reference_dataset)
        io_gdal.match_rasters(reference_dataset, v, tif, ndv=v.no_data_value)
        #Read the resampled tif and extract the array
        ancillarydata[k] = io_gdal.GeoDataset(tif)
        logger.debug('Dataset {} extract.'.format(v))

    if ancillary_failure:
        print('FAILED TO EXTRACT ANCILLARY DATA')
        MPI.COMM_WORLD.Abort(1)
        sys.exit()

    return ancillarydata, dustopacity_band

def main():
    comm = MPI.COMM_WORLD
    rank = comm.rank
    if rank == 0:
        t_start = time.time()

        #Parse the job input
        if len(sys.argv) < 2:
            logger.error("Please supply an input configuration file.")
            sys.exit()
        logger.info("Processing using {} cores".format(comm.size))
        job = io_json.read_json(sys.argv[1])
        #Create a temporary working directory
        workingpath = plio.utils.utils.create_dir(basedir=job['workingdir'])
        # Storage for image / observation parameters
        parameters = {}

        # ISIS and Davinci Processing
        isiscube, parameters = processimage(job, workingpath, parameters)
        parameters = extract_metadata(isiscube, parameters)
        # Extract the temperature data
        temperature = extract_temperature(isiscube)
        if temperature is None:
            logger.error('Failed to extract temperature data.')
            MPI.COMM_WORLD.Abort(1)

        # Check to see if a different reference set has been provided.  If so
        #  update the lat/lon bounds before extraction.

        if "reference" not in job.keys():
            shape =  list(temperature.raster_size)[::-1]
            reference_dataset = temperature
            reference_name = "temperature"
        else:
            in_ref = job["ancillarydata"][job["reference"]]
            logger.info('Using {} as the reference dataset'.format(in_ref))
            # Resample the input reference to the requested resolution
            if "resolution" in job.keys():
                out_ref = os.path.join(workingpath, os.path.splitext(in_ref)[0] + 'resampled.tif')
                xres, yres = job["resolution"]
                opts = gdal.TranslateOptions(xRes=xres, yRes=yres)
                gdal.Translate(out_ref, in_ref, options=opts)
                in_ref = out_ref

            reference_dataset = io_gdal.GeoDataset(in_ref)
            reference_name = job["reference"]
            
            # Get the temperature set to the correct size as well as spatial resolution.
            shape = list(reference_dataset.raster_size)[::-1]
            lower, upper = reference_dataset.latlon_extent
            # Update the start and stop latitudes
            parameters['startlatitude'] = lower[1]
            parameters['stoplatitude'] = upper[1]
            logger.debug('The reference NDV is {}'.format(reference_dataset.no_data_value))
            clipped = os.path.join(workingpath, os.path.splitext(isiscube)[0] + '_clipped.tif')
            io_gdal.match_rasters(reference_dataset, temperature, clipped,
                                  ndv=reference_dataset.no_data_value)
            temperature = extract_temperature(clipped)
        
	    # Extract the ancillary data
        logger.info('Writing temporary files to {}'.format(workingpath))
        ancillarydata, dustopacity_band = extract_ancillary_data(job, temperature, parameters, workingpath, shape, reference_dataset)
        
        t2 = time.time()
        logger.info('Data pre-processing, clipping, and map projection took {} seconds.'.format(t2 - t_start))

        # Extract the lookup table
        hdffile = io_hdf.HDFDataset(job['lookuptables'])

        # All data extracted, so now process
        t1 = time.time()
        eph = interp.EphemerisInterpolator(temperature, ancillarydata, parameters, hdffile)
        eph.interpolate_ephemeris()

        lookup = eph.data
        logger.debug('Lookup table dtype is: {}, with a shape of: {}'.format(lookup.dtype, lookup.shape))
        lookup_shape = lookup.shape
        latslice = eph.get_lat_offsets()
        latitude_nodes = eph.latitudes[latslice]

        t2 = time.time()
        logger.info('Extracting lookup table and season/hour interpolation took {} seconds'.format(t2 - t1))

        #Extract all of the necessary input data arrays
        td_g = temperature.read_array()
        y, x = td_g.shape
        logger.info("Extracting temperature data complete.")

        ndv = temperature.no_data_value
        r_ndv = reference_dataset.no_data_value
        
        logger.debug('The input temperature dataset has a NoDataValue of {}'.format(ndv))
        logger.debug('The reference dataset has a NoDataValue of {}'.format(r_ndv))
        
        #TODO: Magic number here for m to km conversion is poor form.
        if not isinstance(ancillarydata['elevation'], np.ndarray):
            ed_g = ancillarydata['elevation'].read_array() / 1000.0
            #ed_g = ancillarydata['elevation'].read_array()
        else:
            ed_g = ancillarydata['elevation']

        if not isinstance(ancillarydata['slope'], np.ndarray):
            sd_g = ancillarydata['slope'].read_array()
        else:
            sd_g = ancillarydata['slope']

        if not isinstance(ancillarydata['slopeazimuth'], np.ndarray):
            sz_g  = ancillarydata['slopeazimuth'].read_array()
        else:
            sz_g = ancillarydata['slopeazimuth']

        if not isinstance(ancillarydata['albedo'], np.ndarray):
            ad_g = ancillarydata['albedo'].read_array()
        else:
            ad_g = ancillarydata['albedo']

        if not isinstance(ancillarydata['dustopacity'], np.ndarray):
            od_g = ancillarydata['dustopacity'].read_array(band=dustopacity_band)
        else:
            od_g = ancillarydata['dustopacity']

        logger.debug('Min|Max|Mean|Std|Shape')
        names = ['Temperature', 'Elevation', 'Slope', 'SlopeAz', 'Albedo','Tau']
        datas = [td_g,ed_g, sd_g, sz_g, ad_g, od_g ]
        for name, data in zip(names, datas):
            logger.debug('{}: {},{},{},{},{}'.format(name,
                                                     np.min(data),
                                                          np.max(data),
                                                          np.mean(data),
                                                          np.std(data),
                                                          data.shape))


        quotient, remainder = divmod(y, comm.size)
        qrs = (quotient, remainder, td_g.shape)
        temp_fname = temperature.file_name
        logger.debug('Generated all data structures for communication.')

    else:
        #Allocate memory in child processes
        lookup_shape = None
        latitude_nodes = None
        td_g = ed_g = sd_g = sz_g = ad_g = od_g = None
        qrs = None
        ndv = None
        r_ndv = None
        temp_fname = None
        reference_name = None

    #Broadcast scalars
    lookup_shape = comm.bcast(lookup_shape, root=0)
    latitude_nodes = comm.bcast(latitude_nodes, root=0)
    quotient, remainder, shape = comm.bcast(qrs, root=0)
    ndv = comm.bcast(ndv, root=0)
    r_ndv = comm.bcast(r_ndv, root=0)
    temp_fname = comm.bcast(temp_fname, root=0)
    reference_name = comm.bcast(reference_name, root=0)
    comm.Barrier()

    if rank == 0:
        logger.debug('Lookup table shape is {}'.format(lookup_shape))
        logger.debug('Transmitted scalars and temperature geoobject to all cores.')
        tb1 = time.time()

    #Broadcast the ephemeris data (lookup tables)
    if rank != 0:
        lookup = np.empty(lookup_shape, dtype=np.float64)
        temperature = io_gdal.GeoDataset(temp_fname)
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

    reference_lookup = {"temperature":'td',
                        "elevation": 'ed',
                        "slope":'sd',
                        "slopeazimuth":'sz',
                        "albedo":'ad',
                        "dustopacity":'od'}

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

    ref = reference_lookup[reference_name]

    param_interp = interp.ParameterInterpolator(temperature, td, ed, sd,
                                                sz, od, ad, lookup,
                                                latitude_nodes, startpixel,
                                                ndv, r_ndv, ref)

    #Begin the interpolation
    param_interp.bruteforce()
    comm.Barrier()

    if rank != 0:
        result = None
    if rank == 0:
        tb = time.time()
        logger.debug("Parameter interpolation took {} seconds".format(tb-ta))
        # Write out the back planes and then clear the memory so that
        # enough memory is available to perform the gather.

        driver = gdal.GetDriverByName('GTiff')
        bands = 8
        bittype = 'GDT_Float64'
        # x, y, defined above when reading

        outpath = os.path.join(job['outpath'], temperature.base_name)

        logger.debug('Writing to {}.tif'.format(outpath))
        dataset = driver.Create('{}.tif'.format(outpath),
                                x, y, bands, getattr(gdal, bittype))

        # Set the projection and the geotransformation
        dataset.SetGeoTransform(temperature.geotransform)
        projection = temperature.spatial_reference
        if isinstance(projection, str):
            dataset.SetProjection(projection)
        else:
            dataset.SetProjection(projection.ExportToWkt())

        try:
            ndv = temperature.no_data_value
        except:
            ndv = None
        anc_data = [td_g, ed_g, sd_g, sz_g, ad_g, od_g]
        for b in range(2,8):
            bnd = dataset.GetRasterBand(b+1)
            if ndv != None:
                bnd.SetNoDataValue(ndv)
            bnd.WriteArray(anc_data[b-2])
            dataset.FlushCache()
        del anc_data, td_g, ed_g, sd_g, sz_g, ad_g, od_g
        # Allocate space on the master for the result
        result = np.empty((y,x), dtype = np.float32)
        logger.debug(result.shape)
        tc = time.time()
        logger.debug('Initial write took {} seconds'.format(tc - tb))

    # Why is this hitting memory issues on the gather?
    # Now gather the results
    comm.Gatherv([param_interp.resultdata, MPI.FLOAT],
                 [result, scattersize, scatteroffsets, MPI.FLOAT],
                 root=0)

    if rank == 0:
        td = time.time()
        logger.debug("Result gather took {} seconds".format(td - tc))

        # Write the result array
        bnd = dataset.GetRasterBand(1)
        if ndv != None:
            bnd.SetNoDataValue(ndv)
        bnd.WriteArray(result)
        dataset.FlushCache()

        bnd = dataset.GetRasterBand(2)
        qual = np.zeros((y,x))
        if ndv != None:
            bnd.SetNoDataValue(ndv)
            qual[:] = ndv
        else:
            qual[:] = 0
        bnd.WriteArray(np.zeros((y,x)))
        dataset.FlushCache()

        te = time.time()
        logger.debug("Final result write took {} seconds".format(te-td))
        #Cleanup
        logger.debug("Removing the remporary working directory at {}".format(workingpath))
        plio.utils.utils.delete_dir(workingpath)
        logger.info("Processing image {} required {} seconds".format(temperature.base_name, tb - t_start))

if __name__ == '__main__':
    main()
