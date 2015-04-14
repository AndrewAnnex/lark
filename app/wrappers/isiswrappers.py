import logging
import os
import subprocess
import sys

logger = logging.getLogger('ThemisTI')

def checkisisversion():
    """
    Checks that isis3 is in the PATH
    """
    isisversion = os.environ.get('IsisVersion')
    if not isisversion:
        logger.error("ISIS3 must be installed.  Unable to find isis in your environment")
        sys.exit()
    elif isisversion[:5] != 'isis3':
        logger.error("ISIS3 must be installed.  The currently installed version is {}.".format(isisversion))
        sys.exit()
    return

def cam2map(image):
    """
    Wraps the ISIS3 cam2map command.
    """
    logger.info("Running cam2map...")
    basepath, fname = os.path.split(image)
    fname, suffix = os.path.splitext(fname)
    fname += '_proj.cub'
    outimage = os.path.join(basepath, fname)
    cmd = 'cam2map from={} to={} map=$base/templates/maps/simplecylindrical.map'.format(image, outimage)
    cmd = cmd.split()
    response = subprocess.check_output(cmd)
    logger.info("{}".format(response))
    return outimage

def campt(image):
    """
    Wraps the ISIS3 campt command.  Only exposes from variable.

    Parameters
    ----------
    image       str The PATH to the level 0 or level 1 image
    """
    basepath, fname = os.path.split(image)
    fname, suffix = os.path.splitext(fname)

    camptresults = os.path.join(basepath, fname + '_campt.pvl')
    cmd = 'campt from={} to={} format=PVL'.format(image, camptresults)
    cmd = cmd.split()
    response = subprocess.check_output(cmd)
    return response


def spiceinit(image, kernel=None):
    """
    Wraps the ISIS3 spiceinit command

    Parameters
    ----------
    image       str The PATH to the image, this is the from argument
    kernel      str The optional PATH to a user defined ck kernel

    Returns
    -------

    """
    if kernel != None:
        cmd = 'spiceinit from={} ck={}'.format(image, kernel)
        cmd = cmd.split()
        result = subprocess.check_output(cmd)
    else:
        cmd = 'spiceinit from={}'.format(image)
        cmd = cmd.split()
        result = subprocess.check_output(cmd)

    logger.info("{}".format(result))

    return


def thm2isis(image, workingpath):

    basepath, fname = os.path.split(image)
    fname, suffix = os.path.splitext(fname)
    outputimage = os.path.join(workingpath,fname)
    cmd = 'thm2isis from={} to={}'.format(image, outputimage)
    cmd = cmd.split()
    result = subprocess.check_output(cmd)

    logger.info("{}".format(result))

    return outputimage + '.cub'

