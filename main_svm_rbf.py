import numpy as np
import sys
import argparse
from argparse import ArgumentParser
import classification_algo
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

    #C = 2^(-3), 2^(-1), ..., 2^(15)
    #gamma = 2^(-20), ..., 2^(-13)
    # kernel_tuple = ('linear', 'poly', 'rbf', 'sigmoid')
    kernel = 'rbf'
    C_list = np.power(2.0, np.arange(-3, 17, 2))
    gamma_list = np.power(2.0, np.arange(-20, -12))
    degree_list = np.arange(1, 5)
    
    print('-'*60)
    print("SVM")
    if kernel == 'poly':
        for C in C_list:
            for gamma in gamma_list:
                for degree in degree_list:
                    accuracy_list = np.zeros(k_fold)
                    for i in range(k_fold):
                        pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, degree, gamma)
                        accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                        #  print("Run: {}, {}".format(i+1, accuracy))
                        accuracy_list[i] = accuracy
                    print("{0:.3f},kernel={1},C={2},gamma={3},degree={4},SVM".format(accuracy_list.mean(),kernel,C,gamma,degree))
                    #  print("\nAverage accuracy for SVM classifier with "  + kernel + " kernel: {0:.3f}".format(accuracy.mean()))

    elif kernel == 'linear':
        for C in C_list:
            accuracy_list = np.zeros(k_fold)
            for i in range(k_fold):
                pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, None, None)
                accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                #  print("Run: {}, {}".format(i+1, accuracy))
                accuracy_list[i] = accuracy
            print("{0:.3f},kernel={1},C={2},SVM".format(accuracy_list.mean(),kernel,C))
            #  print("\nAverage accuracy for SVM classifier with "  + kernel + " kernel: {0:.3f}".format(accuracy.mean()))
    else:
        for C in C_list:
            for gamma in gamma_list:
                accuracy_list = np.zeros(k_fold)
                for i in range(k_fold):
                    pred = classification_algo.support_vector_machine(X_train[i], y_train[i], X_test[i], C, kernel, None, gamma)
                    accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                    #  print("Run: {}, {}".format(i+1, accuracy))
                    accuracy_list[i] = accuracy
                print("{0:.3f},kernel={1},C={2},gamma={3},SVM".format(accuracy_list.mean(),kernel,C,gamma))

