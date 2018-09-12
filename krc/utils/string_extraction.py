import re

def get_incidence(parsestring):
    """
    Extract the incidence angle from a campt string

    """
    incidence_search = re.compile(r'\s*Incidence', flags=re.I)
    for l in parsestring.splitlines():
        if incidence_search.match(l):
            return float(l.split('=')[1].rstrip())

def get_latitude(parsestring):
    """
    Extract the planetocentric latitude angle from a campt string
    """
    latitude_search = re.compile(r'\s*PlanetocentricLatitude', flags=re.I)
    for l in parsestring.splitlines():
        if latitude_search.match(l):
            return float(l.split('=')[1].rstrip())
