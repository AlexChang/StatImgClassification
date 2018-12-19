from sklearn import svm

param_grid = []

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = svm.SVC()
    else:
        clf = svm.SVC(**hyperParameter.parameterDict)
    return clf