from sklearn import ensemble

def getModel(parameterDict):
    clf = ensemble.AdaBoostClassifier(**parameterDict)
    return clf