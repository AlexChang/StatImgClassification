from sklearn import linear_model

'''
param_grid = [ # logreg_CVGridSearch_accuracy=0.98603_pca=130_max_iter=10000_tol=0.01_C=100_class_weight=balanced_Dec191917.txt
                {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['l2'],
               'dual': [False], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class':['multinomial'],
               'max_iter': [10000]},
               # logreg_CVGridSearch_accuracy=0.98577_pca=130_penalty=l1_max_iter=10000_class_weight=balanced_solver=saga_Dec200934.txt
              {'solver': ['saga'], 'penalty': ['l1'],
               'dual': [False], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['multinomial'],
               'max_iter': [10000]},
               # logreg_CVGridSearch_accuracy=0.98436_pca=130_dual=False_solver=liblinear_tol=0.01_penalty=l1_max_iter=10000_Dec200938.txt
              {'solver': ['liblinear'], 'penalty': ['l1', 'l2'],
               'dual': [False], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]},
                # logreg_CVGridSearch_accuracy=0.98397_pca=130_max_iter=10000_class_weight=None_solver=liblinear_C=1_Dec200954.txt
              {'solver': ['liblinear'], 'penalty': ['l2'], 
               'dual': [True], 'C': [1, 10, 100, 1000], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]} 
                # logreg_CVGridSearch_accuracy=0.98577_pca=130_penalty=l2_dual=False_class_weight=None_tol=0.01_C=0.1_Dec201535.txt
               {'solver': ['newton-cg', 'lbfgs', 'sag', 'saga'], 'penalty': ['l2'],
               'dual': [False], 'C': [0.001, 0.01, 0.1], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class':['multinomial'],
               'max_iter': [10000]},
               # logreg_CVGridSearch_accuracy=0.98474_pca=130_solver=saga_C=0.1_multi_class=multinomial_dual=False_Dec201455.txt
              {'solver': ['saga'], 'penalty': ['l1'],
               'dual': [False], 'C': [0.001, 0.01, 0.1], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['multinomial'],
               'max_iter': [10000]},
               # logreg_CVGridSearch_accuracy=0.98462_pre=none_pca=130_class_weight=None_multi_class=ovr_dual=False_solver=liblinear_Jan162209.txt
              {'solver': ['liblinear'], 'penalty': ['l1', 'l2'],
               'dual': [False], 'C': [0.001, 0.01, 0.1], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]},
               # logreg_CVGridSearch_accuracy=0.98436_pre=none_pca=130_solver=liblinear_C=0.1_class_weight=None_dual=True_Jan162211.txt
              {'solver': ['liblinear'], 'penalty': ['l2'],
               'dual': [True], 'C': [0.001, 0.01, 0.1], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]}]
'''

param_grid = [{'solver': ['liblinear'], 'penalty': ['l2'],
               'dual': [True], 'C': [0.001, 0.01, 0.1], 'tol': [1e-2, 1e-3, 1e-4, 1e-5],
               'class_weight': [None, 'balanced'], 'multi_class': ['ovr'],
               'max_iter': [10000]}]

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = linear_model.LogisticRegression()
    else:
        clf = linear_model.LogisticRegression(**hyperParameter.parameterDict)
    return clf