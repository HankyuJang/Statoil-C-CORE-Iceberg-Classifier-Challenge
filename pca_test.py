import numpy as np
import pandas as pd
import sys
import argparse
from argparse import ArgumentParser
import sklearn.model_selection
import sklearn.metrics
from sklearn.decomposition import PCA

def read_csv(infile):
    # Skip the header
    header = infile.readline().rstrip().split(',')
    x_angle = []
    x_id = []
    X = []
    y = []
    for line in infile:
        line = line.rstrip().split(',')
        # Skip the ID
        x_id.append(line[0])
        x_angle.append(float(line[1]))
        X.append([float(x) for x in line[2:]])
    return np.array(x_id), np.array(x_angle), np.array(X)

if __name__ == '__main__':
    parser = ArgumentParser(description="PCA dimension reduction")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="csv file format with label in the last column", default=sys.stdin)
    parser.add_argument('-o', '--outfile', type=str, help="output file name")
    parser.add_argument('-n', '--num_basis', type=int, default=100, help="number of basis vectors to use for dimension reduction")
    args = parser.parse_args()

    n = args.num_basis
    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X from the input txt file...")
    x_id, x_angle, X = read_csv(args.infile)
    x_id = x_id.reshape((x_id.shape[0],1))
    x_angle = x_angle.reshape((x_angle.shape[0],1))

    print("PCA Dimension Reduction")
    # project the feature space up n dimensions
    pca = PCA(n_components=n)
    pca.fit(X)
    X = pca.transform(X)

    band = ['b' + str(i) for i in range(X.shape[1])]
    
    # Dataset without angle
    # data_wo_angle = np.hstack((x_id, X, y))
    # column_names = ['id'] + band + ['is_iceberg']
    # df_wo = pd.DataFrame(data_wo_angle, columns=column_names)
    # filename = "data/data_processed_pca" + str(n) + "wo_angle.csv"
    # df_wo.to_csv(filename, index=False)

    # Dataset with angle (angle + PCA eigenvectors)
    data_w_angle = np.hstack((x_id, x_angle, X))
    column_names = ['id'] + ['inc_angle'] + band 
    df_w = pd.DataFrame(data_w_angle, columns=column_names)
    filename = "data/" + args.outfile + str(n) + ".csv"
    df_w.to_csv(filename, index=False)

