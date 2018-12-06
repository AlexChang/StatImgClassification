from sklearn import neighbors

param_grid = [{'n_neighbors': list(range(3, 21)), 'weights': ['uniform', 'distance'], 'p': [2, 1],
                 'algorithm': ['auto'], 'leaf_size': [30]}]

def getModel(parameterDict={}):
    clf = neighbors.KNeighborsClassifier(**parameterDict)
    return clf


