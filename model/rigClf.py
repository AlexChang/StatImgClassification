from sklearn import linear_model

'''
param_grid = [{'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
               'alpha': [1e-2, 1e-1, 1, 10, 100], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'max_iter': [10000]}]
'''

param_grid = [{'solver': ['svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga'],
               'alpha': [1e-4, 1e-3, 1000, 10000], 'tol': [1e-4],
               'class_weight': ['balanced'], 'max_iter': [10000]}]

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = linear_model.RidgeClassifier()
    else:
        clf = linear_model.RidgeClassifier(**hyperParameter.parameterDict)
    return clf