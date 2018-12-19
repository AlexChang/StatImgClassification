from sklearn import discriminant_analysis

param_grid = [{'solver': ['svd'], 'shrinkage': [None],
               'store_covariance': [False, True], 'tol': [1e-3, 1e-4, 1e-5]},
              {'solver': ['lsqr', 'eigen'], 'shrinkage': [None, 'auto'],
               'store_covariance': [False, True], 'tol': [1e-3, 1e-4, 1e-5]}]

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = discriminant_analysis.LinearDiscriminantAnalysis()
    else:
        clf = discriminant_analysis.LinearDiscriminantAnalysis(**hyperParameter.parameterDict)
    return clf