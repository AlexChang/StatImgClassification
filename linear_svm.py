from sklearn import svm

def getModel(parameterDict):
    clf = svm.LinearSVC(**parameterDict)
    return clf
