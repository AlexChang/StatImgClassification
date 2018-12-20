from sklearn import linear_model

'''
param_grid = [{'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['l2'],
               'dual': [False], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class':['multinomial'],
               'max_iter': [10000]},
              {'solver': ['saga'], 'penalty': ['l1'],
               'dual': [False], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['multinomial'],
               'max_iter': [10000]},
              {'solver': ['liblinear'], 'penalty': ['l1', 'l2'],
               'dual': [False], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]},
              {'solver': ['liblinear'], 'penalty': ['l2'],
               'dual': [True], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]}]
'''

param_grid = [{'solver': ['liblinear'], 'penalty': ['l2'],
               'dual': [True], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]}]

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = linear_model.LogisticRegression()
    else:
        clf = linear_model.LogisticRegression(**hyperParameter.parameterDict)
    return clf