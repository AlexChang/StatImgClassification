from sklearn import discriminant_analysis

def getModel(parameterDict):
    clf = discriminant_analysis.LinearDiscriminantAnalysis(**parameterDict)
    return clf