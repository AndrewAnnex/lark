import subprocess
import os

def themis_davinci(imagepath, uddw, tesatm, deplaid, rtilt, force, workingpath):
    """
    Calls a processing pipeline script, written in Davinci, to convert from
    a spiceinited cube to calibrated temperature data.

    Parameters
    ----------
    imagepath       str The input PATH to the image to be processed
    uddw            int Davinci script parameter
    tesatm          int Davinci script parameter
    deplaid         int DAvinci script parameter

    Returns
    --------
    outputimage     str PATH to the output image, in ISIS3 cube format

    """

    basepath, fname = os.path.split(imagepath)
    outname, ext = os.path.splitext(fname)
    outpath = os.path.join(workingpath, outname) + '_dvprocessed.cub'
    print imagepath
    print outpath
    cmd = r'./ti_pipeline.dv {} {} {} {} {} {} {}'.format(uddw, tesatm, deplaid, rtilt, force, imagepath, outpath)
    cmd = cmd.split()
    response = subprocess.check_output(cmd)
    print response
    return outpath
