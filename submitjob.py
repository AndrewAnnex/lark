#!/home/jlaura/anaconda/bin/python python
import json

import argparse
import logging
import os
from popen2 import popen2
import sys

def parsearguments():
    """
    Render help and parse the command line arguments

    Parameters
    ----------
    None

    Return
    ------
    args    (class instance)  An argparse class instance
    """

    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', help='Input JSON configuration file to execute the job')
    parser.add_argument('-w', '--walltime', default='01:00:00', dest='walltime', help='Walltime for the job on ths cluster')
    parser.add_argument('-n', '--nodes', default=1, type=int, dest='nnodes', help='The number of nodes to use for processing')
    parser.add_argument('-c', '--cores', default=12, type=int, dest='ncores', help='The number of cores to use per node')

    return parser.parse_args()

def checkisisversion():
    """
    Checks that isis3 is in the PATH
    """
    isisversion = os.environ.get('IsisVersion')
    if not isisversion:
        print "ISIS3 must be installed.  Unable to find isis in your environment"
        sys.exit()
    elif isisversion[:5] != 'isis3':
        print "ISIS3 must be installed.  The currently installed version is {}.".format(isisversion)
        sys.exit()
    return

def validateinputs(parser):
    """
    Validates PATHs and inputs

    Parameters
    ----------
    parser      (instance) An instance of the parser class

    Returns
    -------
    None
    """

    if not os.path.exists(parser.inputfile):
        logger.error( "Unable to locate the input file at {}".format(parser.inlist) )
        sys.exit()
    return

def setuplogger(logfile):
    """
    Setup the logger with both a STDOUT logger that reports info and
    a file logger that reports info and debug information.

    Parameters
    -----------
    logfile     (str) The output path for the log file.

    Returns
    -------
    logger      (instance) The logger object
    """

    logger = logging.getLogger('joblogger')
    logger.setLevel(logging.DEBUG)

    #Log to a file
    flog = logging.FileHandler(logfile)
    flog.setLevel(logging.DEBUG)

    slog = logging.StreamHandler()
    slog.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    flog.setFormatter(formatter)
    slog.setFormatter(formatter)

    logger.addHandler(flog)
    logger.addHandler(slog)

    return logger

def submitjobs(inputfile, walltime, nnodes, ncores, subcmd='qsub'):
    """
    Parse a list of jobs and submit them using qsub

    Parameters
    ----------
    procq       (list) Jobs to iterate
    params      (dict) Parameters
    walltime    (str) The walltime to submit the job with.
    subcmd      (str) Submission command, e.g. qsub or msub

    Returns
    -------
    None
    """
    #Open pipe to the commandline
    outputcl, inputcl = popen2(subcmd)
    pythonpath = '/usgs/apps/anaconda/bin/python'
    command = '/usgs/apps/anaconda/bin/mpirun -n {} --hostfile only_eights.lis {} process.py {} '.format(nnodes * ncores,
                                                                pythonpath,
                                                                inputfile)

    job_string="""#!/bin/bash
#PBS -S /bin/bash
#PBS -N 'ThemisTI_Processing'
#PBS -V
#PBS -o themisti.log
#PBS -e themisti.log
#PBS -l nodes={}:ppn={}
#PBS -l walltime={}
#cd $PBS_O_WORKDIR
cd /home/jlaura/krc_application
{}
""".format(nnodes, ncores, walltime, command)

    inputcl.write(job_string)
    inputcl.close()

    print(job_string)

    job_id = outputcl.read()
    print("Job submitted with ID: {}".format(job_id))

    return

def main():
    parser = parsearguments()
    checkisisversion()
    validateinputs(parser)

    submitjobs(parser.inputfile, parser.walltime, parser.nnodes, parser.ncores)

if __name__ == '__main__':
    main()
