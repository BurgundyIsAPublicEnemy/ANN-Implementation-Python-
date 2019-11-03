import h5py
import numpy as np
import os

cwd = os.getcwd()
xFull = np.empty((0, 49, 2))
yFull = np.empty((0, 49, 2))

for filename in os.listdir(cwd + '/BP4D/2DFeatures/'):
    if filename.endswith(".mat"):
        fh = h5py.File(cwd + '/BP4D/2DFeatures/' + filename, 'r')  # Initializing h5py file handler

        lms_obj = fh.get('fit/pred')  # Extracting the list of landmarks array objects from pred field

        x = (np.zeros((len(lms_obj), 49, 2)))  # Initializing output 3d array

        for i in range(0, len(lms_obj)):  # Iterate over the list to fetch each frame’s landmarks array
            x[i] = fh[lms_obj[i][0]].value.transpose()  # Returns 49*2 numpy array

        labels_obj = fh.get('fit/pose')

        y = np.zeros((len(labels_obj), 49, 2), dtype=('U', '<U1'))

        for i in range(0, len(labels_obj)):  # Iterate over the list to fetch each frame’s landmarks array
            strings = fh[labels_obj[i][0]]
            y[i] = strings  # Returns 49*2 numpy array

        xFull = np.append(xFull, x, axis=0)
        yFull = np.append(yFull, x, axis=0)

print(xFull.shape)
print(yFull.shape)
