import sys
import json
import numpy as np
import pandas as pd
from pandas import DataFrame
import math
# The data is a list of dictionaries
# >>> data[0].keys()
# [u'is_iceberg', u'inc_angle', u'band_2', u'id', u'band_1']
# >>> len(data[0]['band_1'])
# 5625
# >>> len(data[0]['band_2'])
# 5625

# Remove rows if 'is_iceberg' == 'na'
def create_X_y(data):
    X = []
    for line in data:
        if line['inc_angle'] == 'na':
            continue
        # band1 = [float(x) for x in line['band_1']]
        # band2 = [float(x) for x in line['band_2']]
        # band = [math.log(math.exp(elem[0])+math.exp(elem[1])) for elem in zip(band1, band2)]
        band = [math.log(math.exp(float(elem[0]))+math.exp(float(elem[1]))) for elem in zip(line['band_1'], line['band_2'])]
        sub_list = [line['id']]
        sub_list.append(line['inc_angle'])
        sub_list.extend(band)
        # sub_list.extend(line['band_1'])
        # sub_list.extend(line['band_2'])
        sub_list.append(line['is_iceberg'])
        X.append(sub_list)
    return np.array(X)

print("Reading from json file...")
filename = sys.argv[1]
data = json.load(open(filename))

print("Creating X, y...")
X = create_X_y(data)

print("Convert to Pandas DataFrame")
# band_1 = ['b1_' + str(x) for x in range(len(data[0]['band_1']))]
# band_2 = ['b2_' + str(x) for x in range(len(data[0]['band_2']))]
band = ['b' + str(x) for x in range(len(data[0]['band_1']))]
column_names = ['id'] + ['inc_angle'] + band + ['is_iceberg']
df = DataFrame(data=X, columns=column_names)
# >>> df.shape
# (1604, 11252)

##################################################################
# Remove 'inc_angle' column with 133 missing elements
# print("Handling missing data...")
# >>> (df['inc_angle'] == 'na').sum()
# 133

# del df['inc_angle']
# df = df.astype(float)

df.to_csv("data/data_processed_bands_combined.csv", index=False)
