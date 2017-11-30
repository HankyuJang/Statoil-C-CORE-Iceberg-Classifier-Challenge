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
        X.append([float(x) for x in line[2:-1]])
        y.append(int(line[-1]))
    return np.array(x_id), np.array(x_angle), np.array(X), np.array(y)

def read_test(infile):
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
    parser.add_argument('-t', '--testfile', type=argparse.FileType('r'), 
            help="csv file format without label in the last column", default=sys.stdin)
    parser.add_argument('-o', '--outfile', type=str, help="output file name")
    parser.add_argument('-n', '--num_basis', type=int, default=100, help="number of basis vectors to use for dimension reduction")
    args = parser.parse_args()

    n = args.num_basis
    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X and y from the input txt file...")
    train_id, train_angle, X_train, y_train = read_csv(args.infile)
    test_id, test_angle, X_test = read_test(args.testfile)

    train_id = train_id.reshape((train_id.shape[0],1))
    train_angle = train_angle.reshape((train_angle.shape[0],1))
    test_id = test_id.reshape((test_id.shape[0],1))
    test_angle = test_angle.reshape((test_angle.shape[0],1))
    y_train = y_train.reshape((y_train.shape[0],1))

    print("Fitting PCA on training set")
    # project the feature space up n dimensions
    pca = PCA(n_components=n)
    pca.fit(X_train)

    print("Transforming training set")
    X_train = pca.transform(X_train)

    band = ['b' + str(i) for i in range(X_train.shape[1])]
    
    # Dataset without angle
    # data_wo_angle = np.hstack((x_id, X, y))
    # column_names = ['id'] + band + ['is_iceberg']
    # df_wo = pd.DataFrame(data_wo_angle, columns=column_names)
    # filename = "data/data_processed_pca" + str(n) + "wo_angle.csv"
    # df_wo.to_csv(filename, index=False)

    # Dataset with angle (angle + PCA eigenvectors)
    data_w_angle = np.hstack((train_id, train_angle, X_train, y_train))
    column_names = ['id'] + ['inc_angle'] + band + ['is_iceberg']
    df_w = pd.DataFrame(data_w_angle, columns=column_names)
    filename = "data/" + args.outfile + str(n) + ".csv"
    df_w.to_csv(filename, index=False)

    # Testset
    print("Transforming test set")
    X_test = pca.transform(X_test)

    data_w_angle = np.hstack((test_id, test_angle, X_test))
    column_names = ['id'] + ['inc_angle'] + band 
    df_w = pd.DataFrame(data_w_angle, columns=column_names)
    filename = "data/test_" + args.outfile + str(n) + ".csv"
    df_w.to_csv(filename, index=False)

