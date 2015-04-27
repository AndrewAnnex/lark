import h5py as h5
import numpy as np

class HDFDataSet(object):
    """
    Read / Write an HDF5 dataset using h5py
    """

    def __init__(self, filename='/scratch/jlaura/KRC/initialload.h5'):
	self.filename = filename
	self.opendata()

	self.groups = None

    def opendata(self, mode='r'):
	self.data = h5.File(self.filename, mode=mode)

    def getgroups(self):
	"""
	Get all of the first order neighbors to the root node.

	Returns
	-------
	groups : list
		 A unicode list of the keys of the file.
	"""
	if self.groups == None:
	    self.groups = self.data.keys()
	return self.groups

    def getattributes(self):
	if self.groups == None:
	    self.groups = self.data.keys()

	for k in self.groups:
	    print self.data[k].attrs.items()
