import  utils

class HyperParameter(object):

    def __init__(self, method):
        self.parameterDict = {}
        self.method = method

    def addParameter(self, parameterKey, parameterValue):
        self.parameterDict[parameterKey] = parameterValue

    def addParametersByDict(self, parameterDict):
        self.parameterDict.update(parameterDict)

    def loadBestParameter(self):
        parameterDict = utils.loadBestParameters(self.method)
        self.addParametersByDict(parameterDict)

    def toString(self):
        return utils.parameterDictToString(self.parameterDict)
