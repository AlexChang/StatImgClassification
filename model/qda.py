from sklearn import discriminant_analysis

param_grid = []

def getModel(hyperParameter=None):
    if hyperParameter == None:
        clf = discriminant_analysis.QuadraticDiscriminantAnalysis()
    else:
        clf = discriminant_analysis.QuadraticDiscriminantAnalysis(**hyperParameter.parameterDict)
    return clf