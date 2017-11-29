import sys
import json
import numpy as np
import pandas as pd
from pandas import DataFrame

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
        # if line['inc_angle'] == 'na':
            # continue
        sub_list = [line['id']]
        sub_list.append(line['inc_angle'])
        sub_list.extend(line['band_1'])
        sub_list.extend(line['band_2'])
        # sub_list.append(line['is_iceberg'])
        # sub_list.extend([float(x) for x in line['band_1']])
        # sub_list.extend([float(x) for x in line['band_2']])
        # sub_list.append(bool(line['is_iceberg']))
        X.append(sub_list)
    return np.array(X)

print("Reading from json file...")
filename = sys.argv[1]
data = json.load(open(filename))

print("Creating X, y...")
X = create_X_y(data)

print("Convert to Pandas DataFrame")
band_1 = ['b1_' + str(x) for x in range(len(data[0]['band_1']))]
band_2 = ['b2_' + str(x) for x in range(len(data[0]['band_2']))]
column_names = ['id'] + ['inc_angle'] + band_1 + band_2 #+ ['is_iceberg']
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

df.to_csv("data/testdata.csv", index=False)
