from sklearn import neighbors

def getModel(parameterDict):
    clf = neighbors.KNeighborsClassifier(**parameterDict)
    return clf


