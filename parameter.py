class Parameter(object):

    def __init__(self):
        self.parameterDict = {}

    def addParameter(self, parameterKey, parameterValue):
        self.parameterDict[parameterKey] = parameterValue

    def addParametersByDict(self, parameterDict):
        self.parameterDict.update(parameterDict)

    def toString(self):
        parameterString = ""
        for (k, v) in self.parameterDict.items():
            parameterString += str(k) + '=' + str(v) + '_'
        return parameterString
