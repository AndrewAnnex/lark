import argparse
import logging
import os

from plio.utils import log
from krc.wrappers import isiswrapper, pipelinewrapper
from krc.utils.utils import checkdeplaid

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
    parser.add_argument('-l', '--log_level', dest='loglevel', default="INFO", help='The level of logging to use.  Options are: DEBUG, ERROR, DEBUG, INFO')
    parser.add_argument('-k', '--kernel', dest='kernel', default=None, help='An optional kernel for spiceinit')
    parser.add_argument('-u', '--uddw', dest='uddw', default=True, help='Do not apply Davinci UDDW')
    parser.add_argument('-t', '--tesatm', dest='tesatm', default=True, help='Do not apply TES atm. correction')
    parser.add_argument('-r', '--rtilt', dest='rtilt', default=True, help='Do not apply RTilt correction')
    parser.add_argument('-f', '--force', dest='force', default=True, help='Do not apply Force')
    return parser.parse_args()


def main():
    parser = parsearguments()
    log.setup_logging(level=parser.loglevel)
    logger = logging.getLogger(__name__)

    image = parser.inputfile

    logger.info('Reading image {}'.format(image))

    # Get the PATH and the base file name
    basepath, fname = os.path.split(image)
    fname, _ = os.path.splitext(fname)

    workingpath = os.getcwd()
    outcube =  os.path.join(workingpath, '{}.cub'.format(fname))
    print(image, outcube)
    isiswrapper.preprocess_for_davinci(image, outcube, parser.kernel)
    incidence, _, _ = isiswrapper.campt_header(outcube)

    deplaid = checkdeplaid(incidence)
    if deplaid == 'day':
        deplaid = 0
    elif deplaid == 'night':
        deplaid = 1
    else:
        return
    logger.info("If deplaid is set in the input parameters, using {} deplaid routines".format(deplaid))

    dvcube = pipelinewrapper.themis_davinci(image, os.getcwd(), deplaid,
                                            parser.uddw, parser.tesatm,
                                            parser.rtilt, parser.force)

if __name__ == '__main__':
    main()
