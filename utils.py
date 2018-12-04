import parameter
import numpy as np
import csv
import datetime

dataFolder = "data/"
trainDataFileName = "train.csv"
testDataFileName = "test.csv"

resultFolder = "result/"

resultHeader = ['id', 'categories']

trainDataPath = dataFolder + trainDataFileName
testDataPath = dataFolder + testDataFileName

def getTrainData():
    """
    :return: (trainInput, trainOutput): tuple of train input and target
            with shape ([n_samples, n_features], [n_samples])
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
    :return: (testImgId, validTestData): tuple of test image Id and data
            with shape ([n_samples, 1], [n_samples, n_features])
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


def saveResult(result, method='', parameters='', outputFileName=''):
    """
    :param result: np array with shape [n_samples, 2]
    :param method: string (default = '')
    :param parameters: string (default = '')
    :param outputFileName: string (default = '')
    :return:
    """
    header = resultHeader
    timeSuffix = datetime.datetime.now().strftime("%b%d%H%M")
    if outputFileName == '':
        if method != '':
            outputFileName += method + '_'
        if parameters != '':
            outputFileName += parameters
        outputFileName += timeSuffix
    if not outputFileName.endswith('.csv'):
        outputFileName += '.csv'
    outputFilePath = resultFolder + outputFileName
    print("Saving result to : '{}'...".format(outputFilePath))
    f = open(outputFilePath, 'w', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(header)
    for row in result:
        csvWriter.writerow(row)
    f.close()
    print("Save complete!")