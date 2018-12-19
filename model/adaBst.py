from sklearn import ensemble

param_grid = []

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = ensemble.AdaBoostClassifier()
    else:
        clf = ensemble.AdaBoostClassifier(**hyperParameter.parameterDict)
    return clf