from sklearn import neighbors

from parameter import Parameter

class KNNParameter(Parameter):

    def __init__(self, n_neighbors=15):
        super(KNNParameter, self).__init__()
        self.parameterDict['n_neighbors'] = 15

def getModel(parameterDict):
    clf = neighbors.KNeighborsClassifier(**parameterDict)
    return clf


