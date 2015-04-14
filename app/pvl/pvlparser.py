from ast import literal_eval
from collections import OrderedDict
from itertools import islice
import logging
import re

def getheader(fname):

    logger = logging.getLogger('ThemisTI')

    def getfromdict(datadict, maplist):
        return reduce(lambda d, k: d[k], maplist, datadict)

    def setindict(datadict, maplist, value):
        getfromdict(datadict, maplist[:-1])[maplist[-1]] = value

    logger.debug('Extracting header')
    header = OrderedDict()

    fileend = re.compile(r"\bEnd\b", re.IGNORECASE)
    objstart = re.compile(r"\s*\bObject\b")
    objend = re.compile(r"\s*\bEnd_Object\b")
    groupstart = re.compile(r"\s*Group\b")
    groupend = re.compile(r"\s*End_Group\b")

    comment = re.compile(r"\s*#")
    altcomment = re.compile(r"\s*/\*")
    whitespace = re.compile(r"^\s*$")

    commentcounter = 0

    with open(fname, 'rb') as f:
        currentpath = []
        continuation = False

        for i, line in enumerate(f):
            #White space / line break
            if whitespace.match(line):
                continue

            #Header End
            if fileend.match(line):
                logger.debug("Header parsed.")
                break

            #Object
            elif objstart.match(line):
                objkey = re.split('=', line.rstrip())[-1].strip()
                currentpath.append(objkey)
                setindict(header, currentpath, OrderedDict())
            elif objend.match(line):
                currentpath.pop()

            #Group
            elif groupstart.match(line):
                grpkey = re.split('=', line.rstrip())[-1].strip()
                currentpath.append(grpkey)
                setindict(header, currentpath, OrderedDict())

            elif groupend.match(line):
                currentpath.pop()

            elif comment.match(line):
                currentpath.append('Comment{}'.format(commentcounter))
                setindict(header, currentpath, line.strip().rstrip())
                currentpath.pop()
                commentcounter +=1
            elif altcomment.match(line):
                pass
            else:
                #Data
                data = re.split('=', line.rstrip())
                base = data[1].strip().rstrip()
                currentpath.append(data[0].strip())
                if data[1].endswith('-'):
                    continuation = True
                    while continuation:
                        for l in islice(f,1):
                            base = base[:-1] + l.rstrip().strip()
                            if not l.endswith('-'):
                                continuation = False
                elif data[1].endswith(','):
                    continuation = True
                    while continuation:
                        for l in islice(f, 1):
                            base = base + l.rstrip().strip()
                            if l.rstrip().endswith(')'):
                                continuation = False
                try:
                    base = literal_eval(base)
                except:
                    pass
                setindict(header, currentpath, base)
                currentpath.pop()
    return header

def test(inputfile, logger):
    logger.debug('Running test suite using {}'.format(inputfile))

    header = json.dumps(getheader(inputfile, logger), indent=4)
    print header


if __name__ == '__main__':
    import json
    import logging
    import sys

    inputfile = sys.argv[1]

    #Logger to write INFO to the GUI and DEBUG to a file
    logger = logging.getLogger(__name__)
    formatter = logging.Formatter('[%(levelname)s] %(message)s')
    handler = logging.StreamHandler(stream=sys.stdout)
    handler.setFormatter(formatter)
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)


    test(inputfile, logger)
