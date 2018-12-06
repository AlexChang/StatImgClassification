import parameter
import numpy as np
import csv
import json
import datetime
import glob
import joblib

# folders
dataFolder = "data/"
resultFolder = "result/"
modelFolder = "model/"
parameterFolder = "parameter/"
savedModelFolder = "savedModel/"
scoreFolder = "score/"

# data files
trainDataFileName = "train.csv"
testDataFileName = "test.csv"
trainDataPath = dataFolder + trainDataFileName
testDataPath = dataFolder + testDataFileName

# result header
resultHeader = ['id', 'categories']

def loadTrainData():
    """
    :return: (trainInput, trainOutput): tuple of train input and target,
            np array with shape [n_samples, n_features] and [n_samples]
    """
    print("Loading training data from: '{}'...".format(trainDataPath))
    rawTrainData = np.loadtxt(trainDataPath, skiprows=1, delimiter=',')
    validTrainData = rawTrainData[:, 1:]
    trainInput, trainTarget = np.split(validTrainData, (validTrainData.shape[1] - 1, ), axis=1)
    trainTarget = trainTarget.ravel().astype(int)
    print("Load complete!")
    print("Shape of training data: input: {}, target: {}".format(trainInput.shape, trainTarget.shape))
    return (trainInput, trainTarget)


def loadTestData():
    """
    :return: (testImgId, validTestData): tuple of test image Id and data,
            np array with shape [n_samples, 1] and [n_samples, n_features]
    """
    print("Loading test data from: '{}'...".format(testDataPath))
    rawTestData = np.loadtxt(testDataPath, skiprows=1, delimiter=',')
    testImgId = rawTestData[:, :1].astype(int)
    validTestData = rawTestData[:, 1:]
    print("Load complete!")
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

def generateOutputFileName(format, outputFileName='', method='', parameters='', timestamp='', isBest=False):
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
            outputFileName += parameters + '_'
        outputFileName += timestamp
    if isBest:
        outputFileName += '_best'
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


def saveParameters(parameterDict, outputFileName):
    """
    :param clf: classifier
    :param outputFileName: string
    :return:
    """
    header = resultHeader
    outputFilePath = parameterFolder + outputFileName
    print("Saving parameters to : '{}'...".format(outputFilePath))
    f = open(outputFilePath, 'w')
    json.dump(parameterDict, f)
    f.close()
    print("Save complete!")


def loadParameters(inputFileName, format='.json'):
    if not format.startswith('.'):
        format = '.' + format
    if not inputFileName.endswith(format):
        inputFileName += format
    inputFilePath = parameterFolder + inputFileName
    print("Loading model from : '{}'...".format(inputFilePath))
    f = open(inputFilePath, 'r')
    parameterDict = json.load(f)
    f.close()
    print("Load complete!")
    return parameterDict


def loadBestParameters(method, format='.json', isCV=False):
    """
    :param method: string
    :return: parameterDict: dict
    """
    parameterDict = {}
    if not format.startswith('.'):
        format = '.' + format
    if isCV:
        pattern = parameterFolder + method + '*cv*best' + format
    else:
        pattern = parameterFolder + method + '*best' + format
    matchFileList = glob.glob(pattern)
    matchFileList.sort()
    if (len(matchFileList) == 0):
        print("Cannot find parameter file with pattern: '{}'".format(pattern))
        print("Warning: return empty parameter dict!")
    else:
        matchFile = matchFileList[0]
        print("Loading best parameters data from: '{}'...".format(matchFile))
        f = open(matchFile, 'r')
        parameterDict = json.load(f)
        f.close()
        print("Load complete!")
    return parameterDict


def saveModel(clf, outputFileName):
    outputFilePath = savedModelFolder + outputFileName
    print("Saving model to : '{}'...".format(outputFilePath))
    joblib.dump(clf, outputFilePath)
    print("Save complete!")


def loadModel(inputFileName, format='.joblib'):
    if not format.startswith('.'):
        format = '.' + format
    if not inputFileName.endswith(format):
        inputFileName += format
    inputFilePath = savedModelFolder + inputFileName
    print("Loading model from : '{}'...".format(inputFilePath))
    clf = joblib.load(inputFilePath)
    print("Load complete!")
    return clf


def loadBestModel(method, format='.joblib', isCV=False):
    if not format.startswith('.'):
        format = '.' + format
    if isCV:
        pattern = savedModelFolder + method + '*cv*best' + format
    else:
        pattern = savedModelFolder + method + '*best' + format
    matchFileList = glob.glob(pattern)
    matchFileList.sort()
    if (len(matchFileList) == 0):
        raise FileNotFoundError("Cannot find model file with pattern: '{}'".format(pattern))
    else:
        matchFile = matchFileList[0]
        print("Loading best model from: '{}'...".format(matchFile))
        clf = joblib.load(matchFile)
        print("Load complete!")
    return clf


def getScoreResult(clf):
    scoreResult = "Best parameters set found on training set:\n"
    scoreResult += str(clf.best_params_) + "\n"
    scoreResult += "Grid scores on training set:\n"
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        scoreResult += "{:.5f} (+/-{:.5f}) for {}\n" .format(mean, std * 2, str(params))
    return scoreResult


def saveScores(scoreResult, outputFileName):
    outputFilePath = scoreFolder + outputFileName
    print("Saving scores to : '{}'...".format(outputFilePath))
    f = open(outputFilePath, 'w')
    f.write(scoreResult)
    f.close()
    print("Save complete!")
