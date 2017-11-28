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
    parser.add_argument('-activation', '--activation', type=str,
             help="Activation function for the hidden layer. 'identity', no-op activation, useful to implement linear bottleneck, returns f(x) = x 'logistic', the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)). 'tanh', the hyperbolic tan function, returns f(x) = tanh(x). 'relu', the rectified linear unit function, returns f(x) = max(0, x)")
    parser.add_argument('-solver', '--solver', type=str,
             help="The solver for weight optimization. 'lbfgs' is an optimizer in the family of quasi-Newton methods. 'sgd' refers to stochastic gradient descent. 'adam' refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba Note: The default solver 'adam' works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, 'lbfgs' can converge faster and perform better.")
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

    # Use 125 different hidden layers (16, 16, 16) to (64, 64, 64)
    # alpha: 2^(-8) ~ 1
    hls_list = []
    for i in np.power(2, np.arange(4, 7)):
        for j in np.power(2, np.arange(4, 7)):
            for k in np.power(2, np.arange(4, 7)):
                hls_list.append((i, j, k))
    alpha_list = np.power(2.0, np.arange(-8, 1))
    # activation_tuple = ("identity", "logistic", "tanh", "relu")
    # solver_tuple = ("lbfgs", "sgd", "adam")

    #####################################################################
    # Neural Network
    print('-'*60)
    print("\nNeural Network")
    for hls in hls_list:
        for alpha in alpha_list:
            accuracy_list = np.zeros(k_fold)
            for i in range(k_fold):
                pred = classification_algo.neural_network(X_train[i], y_train[i], X_test[i], hls, args.activation, args.solver, alpha)
                accuracy = sklearn.metrics.accuracy_score(y_test[i], pred)
                accuracy_list[i] = accuracy
            print("{0:.3f},hls={1},alpha={2},activation={3},solver={4},NeuralNetwork".format(accuracy_list.mean(),hls,alpha,args.activation,args.solver))
