import glob
import json
import os

template = json.loads("""{
    "images":["/pds_san/PDS_Archive/Mars_Odyssey/THEMIS/data/odtir0_xxxx/i008xxrdr/I00818003RDR.QUB"],
    "latlon":[],
    "uddw": false,
    "tesatm":false,
    "rtilt":true,
    "force":true,
    "name":"THEMIS",
    "projection":{
        "name": "None"
        },
    "resolution":[],
    "outputformat":["GTiff"],
    "bands":[9,10],
    "processing_pipeline":"themis_davinci",
    "ancillarydata":{
        "elevation":"/home/jlaura/Mars_MGS_MOLA_DEM_mosaic_global_463m.cub",
        "slope":"/home/jlaura/Mars_MGS_MOLA_SLOPE_mosaic_global_463m.cub",
        "slopeazimuth":"/home/jlaura/Mars_MGS_MOLA_ASPECT_mosaic_global_463m.cub",
        "albedo":"/home/jlaura/TES_global_Albedo_Christensen_2001.tif",
        "dustopacity":"/home/jlaura/TES_global_Dust_Bandfield_2002.tif",
        "emissivity":1
    }
}""")

files = glob.glob('*.cub')
for f in files:
    base, ext = os.path.splitext(f)
    outname = base + '.json'
    dirname = f.lower()[:4]
    pds_base = "/pds_san/PDS_Archive/Mars_Odyssey/THEMIS/data/odtir0_xxxx/{}xxrdr/{}RDR.QUB".format(dirname, base)
    template['images'] = [pds_base]
    with open(outname, 'w') as out:
        json.dump(template, out)
