# Comparison
This notebook demonstrates using lark to perform comparison between two different versions of the KRC Thermal Model. The validation operates by defining two different version of the KRC Thermal Model. Change cards and model runs are then generated for both of the versions. The output files are then mathematically compared and the differences plotted.

The process is relatively manual in that one must:

  - Modify the params (see the bin directory README for more discussion on the params)
  - Submit the change cards to the cluster and wait while the model runs
  - Generate the plots of differences.
  - Interpret as desired.
