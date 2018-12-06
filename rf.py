from sklearn import ensemble

def getModel(parameterDict):
    clf = ensemble.RandomForestClassifier(**parameterDict)
    return clf