from sklearn import neighbors

param_grid = [{'n_neighbors': list(range(3, 31)), 'weights': ['uniform', 'distance'], 'p': [2, 1],
                 'algorithm': ['auto'], 'leaf_size': [30]}]

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = neighbors.KNeighborsClassifier()
    else:
        clf = neighbors.KNeighborsClassifier(**hyperParameter.parameterDict)
    return clf


