#!/usr/bin/env python3

import numpy as np

# load the npz file
npz_data_file_path = '../data/extrinsics.npz'
npz_data = np.load(npz_data_file_path)

# see what keys are included in the npz file
keys = [key for key in npz_data.files]
print("the keys in file {} are {} \n".format(npz_data_file_path, keys))

# print all key:value pairs in the data set
for key in keys:
    value = npz_data[key]
    print("for key {} we have \n {} \n".format(key, value))

# close the data
npz_data.close()
