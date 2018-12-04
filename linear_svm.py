from sklearn import svm

from parameter import Parameter

class LINEARSVMParameter(Parameter):

    def __init__(self):
        super(LINEARSVMParameter, self).__init__()

def getModel(parameterDict):
    clf = svm.LinearSVC(**parameterDict)
    return clf
