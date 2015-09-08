import json

def read_json(inputfile):
    """
    Read the input json file into a python dictionary.

    Parameters
    ==========
    inputfile : str
                PATH to the file on disk

    Returns
    =======
    jobs : list
           of dictionaries of jobs, one entry per image

    >>> inputs = readinputfile('testfiles/sampleinput.json')
    >>> k = inputs.keys()
    >>> k.sort()
    >>> print k
    [u'ancillarydata', u'bands', u'force', u'images', u'latlon', u'name', u'outputformat', u'processing_pipeline', u'projection', u'resolution', u'rtilt', u'tesatm', u'uddw']

    """
    with open(inputfile, 'r') as f:
	try:
            jdict = json.load(f)
	except:
	    raise IOError
    return jdict
