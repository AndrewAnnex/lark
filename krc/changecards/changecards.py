import itertools
import os
import sys
import numpy as np

from krc.utils.utils import computeinertia

def extractdata(infile, datastart):
    """
    Parameters
    ----------
    infile : str
             Input file PATH
    datastart : int
                The binary offset to start reading at
    Returns
    -------
           : ndarray
             n x 37 x (nseason * 2) array
    """
    with open(infile, 'rb') as f:
        f.seek(datastart)
        d = np.fromfile(f, dtype=np.float64).reshape((96, 37, -1), order='F')
        return d[:48, :, ::2]
        #d = d[:d.shape[0]/2]#.reshape(96, 37, -1)
        #d = d.reshape(96, 37, -1)
    
def getfiles(indir, parameters, version, ext='.tds', log_inertias=True, nlats=37, nhours=48, byfname=False):
    # Parse albedo
    nalb = len(parameters['albedo'])
    albs = [i for i in parameters['albedo']]

    # Parse the inertia into an exponential range
    if log_inertias:
        ninertia = parameters['inertia'][2]
        inertias = computeinertia(parameters['inertia']).tolist()
    else:
        inertias = parameters['inertia']
        ninertia = len(inertias)

    # Parse Tau
    tau = parameters['tau']
    ntau = len(tau)
    taus = [i for i in tau]

    # Parse elevations
    elev = parameters['elevation']
    nelev = len(elev)
    elevstr = [i for i in elev]

    # Parse emissivity
    emissivity = parameters['emissivity']
    nemiss = len(emissivity)
    emissivities = [i for i in emissivity]

    # Parse SlopeAZ
    slopeaz = parameters['slope_azimuth']
    nslopeaz = len(slopeaz)
    slopeazs = [i for i in slopeaz]

    # Parse Slopes
    slope = parameters['slope']
    nslope = len(slope)
    slopes = [i for i in slope]

    print("Reading {} files.".format(
        nalb * ninertia * ntau * nemiss * nslopeaz * nslope * nelev))
    
    combinations = list(itertools.product(inertias, slopes, slopeazs, elevstr, emissivities, taus, albs))

    seasons = {}
    if byfname == False:
        for i in range(80):
            seasons[i] = np.empty((48,37,270))
    cnt = 0
    for c in combinations:
        if c[3] < 0:
            el = 'em' + str(abs(int(c[3])))
        else:
            el = 'ep' + str(abs(int(c[3])))
        fname = 'sk{0:04d}a{1:03d}t{2:03d}es{3:03d}{4}az{5:03d}sl{6:03d}{7}'.format(int(c[0]),
                                                                                  int(c[6] * 100),
                                                                                  int(c[5] * 100),
                                                                                  int(c[4]* 100),
                                                                                  el,
                                                                                  int(c[2]),
                                                                                  int(c[1]), ext)
        if version == 321:
            multi = 32
        else:
            multi = 16
        datastart = nlats * nhours * multi
        data = extractdata(os.path.join(indir, fname), datastart)

        if byfname == True:
            seasons[fname] = data
        else:
            for i in range(80):
                seasons[i][:,:,cnt] = data[:,:,i]
            cnt += 1
    return seasons


def createchangecards(header, parameters, outfile, outdir, albstr='a',
                      inertiastr='sk', taustr='t', log_inertias=True,
                      log='changecards.lis', npartitions=20):

    # Parse albedo
    nalb = len(parameters['albedo'])
    albs = [i for i in parameters['albedo']]

    # Parse the inertia into an exponential range
    if log_inertias:
        ninertia = parameters['inertia'][2]
        inertias = computeinertia(parameters['inertia']).tolist()
    else:
        inertias = parameters['inertia']
        ninertia = len(inertias)

    # Parse Tau
    tau = parameters['tau']
    ntau = len(tau)
    taus = [i for i in tau]

    # Parse elevations
    elev = parameters['elevation']
    nelev = len(elev)
    elevstr = [i for i in elev]

    # Parse emissivity
    emissivity = parameters['emissivity']
    nemiss = len(emissivity)
    emissivities = [i for i in emissivity]

    # Parse SlopeAZ
    slopeaz = parameters['slope_azimuth']
    nslopeaz = len(slopeaz)
    slopeazs = [i for i in slopeaz]

    # Parse Slopes
    slope = parameters['slope']
    nslope = len(slope)
    slopes = [i for i in slope]

    print("Generating {} files for {} elevations".format(
        nalb * ninertia * ntau * nemiss * nslopeaz * nslope, nelev))

    # Get the output ext and the output directory
    ext = '.tds'

    parameters = list(itertools.product(
        inertias, albs, taus, emissivities, slopeazs, slopes))
    chunksize = int(len(parameters) / npartitions)
    chunks = [parameters[x:x + chunksize]
              for x in range(0, len(parameters), chunksize)]

    file_names = []
    # Loop over the elevations
    for e, elevation in enumerate(elev):
        elevationstr = """"""
        for j, l in enumerate([10, 10, 10, 7]):
            elevations = [elevation for v in range(l)]
            if j != 3:
                elevationstr += ''.join(str(w).rjust(7)
                                        for w in elevations) + '\n'
            else:
                elevationstr += ''.join(str(w).rjust(7) for w in elevations)

        for fnum, chunk in enumerate(chunks):
            ename = elev[e]
            if ename < 0:
                ename = 'n{}'.format(abs(ename))
            cfile = '{}_{}_{}.inp'.format(outfile, ename, fnum)
            with open(cfile, 'w') as out:
                out.write(header.format(elevationstr))
                if elevation < 0:
                    elevstr = 'em'
                else:
                    elevstr = 'ep'
                fnameelev = elevstr + str(abs(int(elevation)))

                for c in chunk:
                    inertia = c[0]
                    albedo = c[1]
                    tau = c[2]
                    emiss = c[3]
                    slopeaz = c[4]
                    slope = c[5]

                    # Albedo
                    out.write("1 1 {} 'Albedo'\n".format(round(float(albedo), 2)))
                    fnamealbedo = albstr + "{0:03d}".format(int(albedo * 100))

                    # Tau
                    out.write("1 17 {} 'Tau dust'\n".format(round(float(tau), 2)))
                    fnametau = taustr + "{0:03d}".format(int(tau * 100))

                    # Emissivity
                    out.write("1 2 {} 'Emissivity'\n".format(
                        round(float(emiss), 2)))
                    fnameemis = 'es{0:03d}'.format(int(emiss * 100))

                    # Slope Az
                    out.write("1 24 {} 'Slope Azimuth'\n".format(
                        round(float(slopeaz), 2)))
                    fnameslopeaz = 'az{0:03d}'.format(int(slopeaz))

                    # Slope
                    out.write("1 23 {} 'Slope'\n".format(round(float(slope), 2)))
                    fnameslope = 'sl{0:03d}'.format(int(slope))

                    # Inertia
                    out.write("1 3 {} 'Inertia'\n".format(
                        round(float(inertia), 1)))
                    fnameinertia = inertiastr + "{0:04d}".format(int(inertia))

                    # Write out the file PATH
                    outpath = os.path.join(outdir, fnameinertia + fnamealbedo +
                                        fnametau + fnameemis + fnameelev + fnameslopeaz + fnameslope + ext)
                    file_names.append(outpath)
                    if len(outpath) > 90:
                        print("Error!  Total filename length (path + name) exceeds 90 characters (Fortran limitation)")
                        print("Please select a new path with less nesting.")
                        sys.exit()
                    out.write("8 21 0 '{}'\n".format(outpath))
                    out.write('0/\n')
                for i in range(3):
                    out.write('0/\n')
