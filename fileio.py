#!/usr/bin/env python

import json

def readinputfile(inputfile):
    """
    Read the input json file into a python dictionary.

    Parameters
    ==========
    inputfile   str PATH to the file on disk

    Returns
    =======
    jobs        list of dictionaries of jobs, one entry per image

    >>> inputs = readinputfile('testfiles/sampleinput.json')
    >>> k = inputs.keys()
    >>> k.sort()
    >>> print k
    [u'bands', u'images', u'latlon', u'outputformat', u'projection', u'rtilt', u'tesatm', u'uddw']

    """
    with open(inputfile, 'r') as f:
        jobjson = json.load(f)

    return jobjson

if __name__ == "__main__":
    import doctest
    doctest.testmod()
