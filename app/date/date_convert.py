import datetime
from app.config import DATEFMT

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
