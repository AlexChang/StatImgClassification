from sklearn import svm

'''
param_grid = [{'penalty': ['l2'], 'loss': ['squared_hinge', 'hinge'], 'dual': [True],
                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'max_iter': [10000]},
                {'penalty': ['l1', 'l2'], 'loss': ['squared_hinge'], 'dual': [False],
                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'max_iter': [10000]}]
'''

param_grid = [{'penalty': ['l2'], 'loss': ['squared_hinge', 'hinge'], 'dual': [True],
                 'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5], 'max_iter': [10000]}]

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = svm.LinearSVC()
    else:
        clf = svm.LinearSVC(**hyperParameter.parameterDict)
    return clf