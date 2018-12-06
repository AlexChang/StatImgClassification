from sklearn import neighbors

param_grid = [{'penalty': ['l2'], 'loss': ['squared_hinge', 'hinge'], 'dual': [True],
                     'C': [1, 10, 100, 1000], 'tol': [1e-3, 1e-4, 1e-5], 'max_iter': [5000]},
                    {'penalty': ['l1', 'l2'], 'loss': ['squared_hinge'], 'dual': [False],
                     'C': [1, 10, 100, 1000], 'tol': [1e-3, 1e-4, 1e-5], 'max_iter': [5000]},]

def getModel(parameterDict={}):
    clf = neighbors.KNeighborsClassifier(**parameterDict)
    return clf


