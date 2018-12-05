from sklearn import discriminant_analysis

def getModel(parameterDict):
    clf = discriminant_analysis.QuadraticDiscriminantAnalysis(**parameterDict)
    return clf