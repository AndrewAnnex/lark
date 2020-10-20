### KRC Thermal Model Interpolation Application

This code base is designed to use output from the KRC thermal model along with multiple other input data sets in order to model thermal inertia of the Martian surface.

Model inputs:

  - [KRC Thermal Model](https://github.com/USGS-Astrogeology/krc_thermal_model) lookup tables merged into HDF5 format.
  - Slope - An arbitrary slope map in a geospatially enabled format
  - Slope Azimuth - An arbitrary slope azimuth map in a geospatially enabled format. Normally derived alongside the slope map though this is not required.
  - Elevation - An arbitrary elevation data set. This is height above (or below) the ellipsoid in km.
  - Albedo - An aledbo map, e.g., Christensen 2001
  - Emissivity - A constant emissivity.
  - Spatial location (lat, lon) - derived from the input THEMIS images.
  - Mars Ls - The [martian solar longitude](http://www-mars.lmd.jussieu.fr/mars/time/solar_longitude.html).
  - THEMIS band 9 observed temperature.
  
Using the inputs, this code base uses the KRC Thermal model as truth and then estimates the observed thermal inertia for every pixel within the THEMIS image. We use an interpolation strategy so that we can run a limited number of KRC Thermal Model iterations (which arguably would not scale to a full THEMIS image at full resolution) and then use those as nodes for subsequent interpolation.

## Configuration

We use a configuration file to manage model iterations. A sample configuration file is reproduced below:

```
{"name": "THEMIS",
    "tesatm": true,
    "processing_pipeline": "themis_davinci",
    "uddw": true,
    "outputformat": ["GTiff"],
    "workingdir": "/scratch/jlaura/Mars2020_output",
    "outpath": "/scratch/jlaura/Mars2020_output",
    "latlon": [],
    "force": true,
    "images": "/pds_san/PDS_Archive/Mars_Odyssey/THEMIS/USA_NASA_PDS_ODTSDP_100XX/ODTSDP_10040/data/odtir1_0040/i441xxrdr/I44149002RDR.QUB",
    "ancillarydata": {"slope": 0.0,
                      "dustopacity": "montone",
                      "slopeazimuth": 0.0,
                      "elevation": "/scratch/rfergason/ti_algorithm/input_final/elevation_MOLA_HRSC/Mars_MOLA_blend200ppx_HRSC_DEM_clon0dd_200mpp_lzw.tif",
                      "albedo": "/scratch/jlaura/KRC/basemaps/TES_global_Albedo_Christensen_2001.tif",
                      "emissivity": 1.0},
    "rtilt": true,
    "lookuptables": "/scratch/jlaura/krc_lookup344.h5",
    "projection": {"name": "None"},
    "deplaid": false,
    "resolution": [],
    "bands": [9, 10]}
    ```


