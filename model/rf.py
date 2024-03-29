from sklearn import ensemble

param_grid = []

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = ensemble.RandomForestClassifier()
    else:
        clf = ensemble.RandomForestClassifier(**hyperParameter.parameterDict)
    return clf