import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.model_selection
import sklearn.metrics

def read_csv(infile):
    # Skip the header
    header = infile.readline().rstrip().split(',')
    X = []
    y = []
    for line in infile:
        line = line.rstrip().split(',')
        # Skip the ID
        X.append([float(x) for x in line[1:-1]])
        y.append(int(line[-1]))
    return np.array(X), np.array(y)

def prepare_train_test_using_kfold(k, X, y):
    trainset, testset = [], []
    kf = sklearn.model_selection.KFold(n_splits=k, shuffle=True)
    kf.get_n_splits(X)
    X_train = []
    y_train = []
    X_test = []
    y_test = []
    print(kf)
    for train_index, test_index in kf.split(X):
        X_train.append(X[train_index])
        y_train.append(y[train_index])
        X_test.append(X[test_index])
        y_test.append(y[test_index])
    return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    parser = ArgumentParser(description="Read csv file and create a dataset k-fold cross validation")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="csv file format with label in the last column", default=sys.stdin)
    parser.add_argument('-o', '--outfile', help="output file name")
    parser.add_argument('-kfold', '--kfold', type=int, default=5, help="k-fold (positive integer. default is 5-fold)")
    args = parser.parse_args()

    k_fold = args.kfold
    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X and y from the input txt file...")
    X, y = read_csv(args.infile)

    #####################################################################
    # Get trainsets and testsets using K-Fold
    print("Preparing the trainsets and testsets using {} fold...".format(k_fold))
    X_train, y_train, X_test, y_test = prepare_train_test_using_kfold(k_fold, X, y)

    np.savez(args.outfile, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, Kfold = k_fold)
    print("-"*60)
    print("trainset and testset are created!")
    print("They are saved as `dataset.npz` in the same directory")

