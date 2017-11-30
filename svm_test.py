import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import sklearn.metrics
from sklearn.svm import SVC
import pandas as pd

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

def read_test(infile):
    # Skip the header
    header = infile.readline().rstrip().split(',')
    X = []
    id_list = []
    for line in infile:
        line = line.rstrip().split(',')
        # Skip the ID
        id_list.append(line[0])
        X.append([float(x) for x in line[1:]])
    return np.array(id_list), np.array(X)


if __name__ == '__main__':
    parser = ArgumentParser(description="Training, Testing...")
    parser.add_argument('-i', '--infile', type=argparse.FileType('r'), 
            help="csv file format with label in the last column", default=sys.stdin)
    parser.add_argument('-t', '--testfile', type=argparse.FileType('r'), 
            help="csv file format without label in the last column", default=sys.stdin)
    parser.add_argument('-C', '--C', type=float,
             help="Penalty parameter C of the error term.")
    parser.add_argument('-kernel', '--kernel', type=str,
             help="Specifies the kernel type to be used in the algorithm. It must be one of 'linear', 'poly', 'rbf', 'sigmoid', 'precomputed' or a callable. If none is given, 'rbf' will be used. If a callable is given it is used to pre-compute the kernel matrix from data matrices; that matrix should be an array of shape (n_samples, n_samples)")
    parser.add_argument('-degree', '--degree', type=int, default=1,
             help="Degree of the polynomial kernel function ('poly'). Ignored by all other kernels.")
    parser.add_argument('-gamma', '--gamma', type=float,
             help="Kernel coefficient for 'rbf', 'poly' and 'sigmoid'. If gamma is 'auto' then 1/n_features will be used instead.")
    args = parser.parse_args()

    #####################################################################
    # Prepare X and y from the input txt file
    print("Preparing X_train and y_train from the input csv file...")
    X_train, y_train = read_csv(args.infile)
    print("Preparing X_test from the test csv file...")
    id_list, X_test = read_test(args.testfile)

    kernel = args.kernel
    C = args.C
    gamma = args.gamma
    degree = int(args.degree)
    
    clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, probability=True)

    print("Training the model")
    clf.fit(X_train, y_train)

    print("Testing the testdata")
    # pred = clf.predict(X_test)
    pred = clf.predict_proba(X_test)

    output = np.hstack((id_list.reshape((id_list.shape[0], 1)), pred[:,1].reshape((pred[:,1].shape[0], 1))))
    
    df = pd.DataFrame(output, columns = ["id", "is_iceberg"])
    df.to_csv("data/output.csv", index=False)
