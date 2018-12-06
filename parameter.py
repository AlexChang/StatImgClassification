import  utils

class Parameter(object):

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
        parameterString = ""
        for (k, v) in self.parameterDict.items():
            parameterString += str(k) + '=' + str(v) + '_'
            if len(parameterString) > 100:
                break
        if parameterString.endswith('_'):
            parameterString = parameterString[:-1]
        return parameterString
