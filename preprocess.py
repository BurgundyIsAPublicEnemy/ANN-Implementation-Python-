import h5py, os, sys
import numpy as np
import csv
import scipy.io

class preprocess():
	def run(self):
		#Load .mat files for points
		cwd = os.getcwd()
		print(cwd)
		fh = h5py.File(cwd + '/BP4D/2DFeatures/F001_T1.mat', 'r')  # Initializing h5py file handler
		lms_obj = fh.get('fit/pred') # Extracting the list of landmarks array objects from pred field
		all_points_array = np.zeros((len(lms_obj), 49, 2)) # Initializing output 3d array
		for i in range(0, len(lms_obj)): # Iterate over the list to fetch each frame’s landmarks array
			all_points_array[i] = fh[lms_obj[i][0]].value.transpose() # Returns 49*2 numpy array

		#Load .csv files for labels
		cwd = os.getcwd()
		print(cwd)

		all_labels_array = [ ]
		with open(cwd + '/BP4D/Labels/OCC/F001_T1.csv') as f:
		    reader = csv.reader(f, delimiter=',')
		    next(reader, None)
		    for row in reader:
		        rowall_labels_array = [ float(elem) for elem in row ]
		        all_labels_array.append(rowall_labels_array)

		all_labels_array = np.asarray(all_labels_array)

		#conv (6) to (49,2)
		all_points_array = all_points_array[0:len(all_labels_array)]
		return(all_points_array, all_labels_array)

if __name__ == "__main__":
	preprocess().run()
