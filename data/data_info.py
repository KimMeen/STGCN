# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 14:56:50 2020

@author: Ming Jin
"""

import numpy as np
import pandas as pd

file_path = './METR-LA/metr-la.h5'

df = pd.read_hdf(file_path)

num_samples, num_attributes = df.shape
print("\nnum_samples:", num_samples)
print("num_attributes:", num_attributes, "\n")
print(df)