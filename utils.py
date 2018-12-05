import parameter
import numpy as np
import csv
import json
import datetime
import glob

dataFolder = "data/"
trainDataFileName = "train.csv"
testDataFileName = "test.csv"

resultFolder = "result/"

parameterFoler = "parameter/"

resultHeader = ['id', 'categories']

trainDataPath = dataFolder + trainDataFileName
testDataPath = dataFolder + testDataFileName

def getTrainData():
    """
    :return: (trainInput, trainOutput): tuple of train input and target,
            np array with shape [n_samples, n_features] and [n_samples]
    """
    print("Reading training data from: '{}'...".format(trainDataPath))
    rawTrainData = np.loadtxt(trainDataPath, skiprows=1, delimiter=',')
    validTrainData = rawTrainData[:, 1:]
    trainInput, trainTarget = np.split(validTrainData, (validTrainData.shape[1] - 1, ), axis=1)
    trainTarget = trainTarget.ravel().astype(int)
    print("Read complete!")
    print("Shape of training data: input: {}, target: {}".format(trainInput.shape, trainTarget.shape))
    return (trainInput, trainTarget)


def getTestData():
    """
    :return: (testImgId, validTestData): tuple of test image Id and data,
            np array with shape [n_samples, 1] and [n_samples, n_features]
    """
    print("Reading test data from: '{}'...".format(testDataPath))
    rawTestData = np.loadtxt(testDataPath, skiprows=1, delimiter=',')
    testImgId = rawTestData[:, :1].astype(int)
    validTestData = rawTestData[:, 1:]
    print("Read complete!")
    print("Shape of test data: imgId: {}, input: {}".format(testImgId.shape, validTestData.shape))
    return (testImgId, validTestData)


def concatenateResult(testImgId, predictionResult):
    """
    :param testImgId: np array with shape [n_samples, 1]
    :param predictionResult: np array with shape [n_samples, 1]
    :return: result: np array with shape [n_samples, 2]
    """
    result = np.concatenate((testImgId, predictionResult), axis=1)
    result = result.astype(int)
    return result

def generateOutputFileName(format, outputFileName='', method='', parameters='', timestamp=''):
    """
    :param format: file format, string (like 'csv', 'json')
    :param outputFileName: string (default = '')
    :param method: string (default = '')
    :param parameters: string (default = '')
    :param timestamp: string (default = '')
    :return:
    """
    if not format.startswith('.'):
        format = '.' + format
    if timestamp == '':
        timestamp = datetime.datetime.now().strftime("%b%d%H%M")
    if outputFileName == '':
        if method != '':
            outputFileName += method + '_'
        if parameters != '':
            outputFileName += parameters
        outputFileName += timestamp
    if not outputFileName.endswith(format):
        outputFileName += format
    return outputFileName


def saveResult(result, outputFileName):
    """
    :param result: np array with shape [n_samples, 2]
    :param outputFileName: string
    :return:
    """
    header = resultHeader
    outputFilePath = resultFolder + outputFileName
    print("Saving result to : '{}'...".format(outputFilePath))
    f = open(outputFilePath, 'w', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(header)
    for row in result:
        csvWriter.writerow(row)
    f.close()
    print("Save complete!")


def saveParameters(clf, outputFileName):
    """
    :param clf: classifier
    :param outputFileName: string
    :return:
    """
    header = resultHeader
    outputFilePath = parameterFoler + outputFileName
    print("Saving parameters to : '{}'...".format(outputFilePath))
    f = open(outputFilePath, 'w')
    json.dump(clf.__dict__, f)
    f.close()
    print("Save complete!")


def getBestParameters(method):
    """
    :param method: string
    :return: parameterDict: dict
    """
    parameterDict = {}
    pattern = parameterFoler + method + '*best.json'
    matchFileList = glob.glob(pattern)
    matchFileList.sort()
    if (len(matchFileList) == 0):
        print("Cannot find parameter file with pattern: '{}'".format(pattern))
    else:
        matchFile = matchFileList[0]
        print("Reading best parameters data from: '{}'...".format(matchFile))
        f = open(matchFile, 'r')
        parameterDict = json.load(f)
        f.close()
        print("Read complete!")
    print("Best parameters: {}".format(parameterDict))
    return parameterDict