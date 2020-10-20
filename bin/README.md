# Scripts
The scripts in this directory are installed when one installs the lark library.

## Generating the KRC Thermal Model Output
The KRC Thermal Model is required as an input. We supply tools to build the input HDF5 files.

### Create change cards for the KRC Thermal Model
This script is used to generate a large number of KRC Thermal Model change cards.
Usage: `krc_create_changecards`
This script generates the change cards and accepts the following arguments:

  * `-c` A JSON configuration file with the desired changecard parameters.  The default is 'parameters.config'.
  * `-c` The desired file extension.  The default is '.tds'.
  * `-m` A flag to generate Mellon, instead of Kieffer tables.
  * `outfilename` (required) The name of the output file(s).  Additional information is added to ensure unique naming.
  * `outdir` (required) The output directory where the lookup tables are to be written.
  
The `parameters.config` file defines the nodes at which the model will be computed. An example configuration is provided below:

```
{
    "albedo": [0.08, 0.22, 0.32],
    "slope_azimuth": [0, 75, 210, 285,360],
    "slope": [0.0, 15.0, 30.0],
    "tau": [0.02, 0.30, 0.62],
    "elevation": [-5.0, -2.0, -1.0, 1.0, 6.0, 8.0],
    "inertia": [24, 3000, 20],
    "emissivity": [0.85, 0.90, 1.00]
}
```

  * This is a  standard JSON file will all of the desired nodes enumerated.
  * Note that inertia is defined as [start value, stop value, step size].
  
 ### Submit the change cards
 We submit the change cards to a slurm based cluster to generate them in parallel. Since, in our uses, we generate 48000+ model runs, we need to use the cluster to make this computationally efficient.
 Usage: `krc_submit_changecards'
 
   * `-x` The extension of the change card files. By default this is `.inp`.
   * `-v` Version of krc being used. The default is `355` for version `3.5.5`.
   * `input_dir` (required) - This is the directory containing the change cards.
   * `working dir` (required) - The PATH to the working directory where outputs are written.
   * `binary` (required) - The PATH to the KRC binary that will process the change cards.
   
### Spawn the KRC jobs
This directory contains the `krc_spawnkrc` script. This script is used internally only to manage the interaction between a cluster job and the krc thermal model.

### Load the HDF5 file
Once some number of KRC thermal model output files have been generated, we load them into JDF5 for more efficient input/output.
Usage: `krc_loadhdf`

  * `-x` - The extension of the KRC Thermal model output. The default is `.tds`
  * `-c` - The configuration file that identifies the nodes. See above for the default, `parameters.config`.
  * `-a` - The abbreviation used for albedo in the file name. The default is `a`.
  * `-i` - The abbreviation used for the albedo string in the file name. The default is `sk`.
  * `-t` - The abbreviation used for the tau string in the file name. The default is `t`.
  * `-v` - The KRC version that was used to create the lookup tables.
  * `-n` - The number of latitudes in each KRC Thermal Model output file. The default is 37.
  * `-d` - The number of hours in each KRC Thermal Model output file. The default is 48.
  * `-s` - The number of seasons in each KRC file. The default is 80.
  * `tabledir` (required) - The directory containg the output of the KRC model.
  * `outputfile` (required) - The location of the output file.
  
# Submitting an iterpolation run
The `krc_submit_interpolation` file is used to submit one run of the lark model.
Usage: `krc_submit_interpolation`

  * `inputfile` - the configuration.json file used to parametrize the run
  * `-w` - The walltime
  * `-n` - The number of nodes to use
  * `-c` - The number of cores per node to use
  * `-f` - An optional file list. Normally, this code runs on a single file, as defined in the input. As an alternative, if a filelist is passed, all files in the filelist are processed sharing the parameters in the `inputfile`.
  * `-m` - The manager to use. Slurm or PBS.
