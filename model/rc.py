from sklearn import linear_model

param_grid = [{'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
               'alpha': [1e-2, 1e-1, 1, 10, 100], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'max_iter': [10000]}]

def getModel(parameterDict={}):
    clf = linear_model.RidgeClassifier(**parameterDict)
    return clf