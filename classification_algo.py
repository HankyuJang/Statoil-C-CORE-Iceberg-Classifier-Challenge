import sklearn.svm
import sklearn.ensemble
import sklearn.neighbors
import sklearn.neural_network

def support_vector_machine(X_train, y_train, X_test, C, kernel, degree, gamma):
    if C==None:
        C=1.0
    if kernel==None:
        kernel="rbf"
    if degree==None:
        degree=3
    if gamma==None:
        gamma='auto'
    clf = sklearn.svm.SVC(C=C, kernel=kernel, degree=degree, gamma=gamma)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def random_forest(X_train, y_train, X_test, n, criterion, minss):
    if n==None:
        n=10
    if criterion==None:
        criterion="gini"
    if minss==None:
        minss=2
    clf = sklearn.ensemble.RandomForestClassifier(n_estimators=n, criterion=criterion, min_samples_split=minss)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)

def k_nearest_neighbor(X_train, y_train, X_test, n, weights):
    if n==None:
        n=5
    if weights==None:
        weights="uniform"
    neigh = sklearn.neighbors.KNeighborsClassifier(n_neighbors=n, weights=weights)
    neigh.fit(X_train, y_train)
    return neigh.predict(X_test)
    
def neural_network(X_train, y_train, X_test, hls, activation, solver, alpha): 
    if hls==None:
        hls=(100,)
    if activation==None:
        activation="relu"
    if solver==None:
        solver="adam"
    if alpha==None:
        alpha=0.0001
    clf = sklearn.neural_network.MLPClassifier(hidden_layer_sizes=hls, activation=activation, solver=solver, alpha=alpha)
    clf.fit(X_train, y_train)
    return clf.predict(X_test)
