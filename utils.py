import parameters
import numpy as np
import csv
import datetime

## (trainInput, trainOutput)
## ([sample x feature], [sample x 1])
def getTrainData():
    print("Reading training data from: '{}'...".format(parameters.trainDataPath))
    rawTrainData = np.loadtxt(parameters.trainDataPath, skiprows=1, delimiter=',')
    validTrainData = rawTrainData[:, 1:]
    trainInput, trainTarget = np.split(validTrainData, (validTrainData.shape[1] - 1, ), axis=1)
    print("Read complete!")
    print("Shape of training data: input: {}, target: {}".format(trainInput.shape, trainTarget.shape))
    return (trainInput, trainTarget)


## (testImgId, validTestData)
## ([sample x 1], [sample x feature])
def getTestData():
    print("Reading test data from: '{}'...".format(parameters.testDataPath))
    rawTestData = np.loadtxt(parameters.testDataPath, skiprows=1, delimiter=',')
    testImgId = rawTestData[:, :1]
    validTestData = rawTestData[:, 1:]
    print("Read complete!")
    print("Shape of test data: imgId: {}, input: {}".format(testImgId.shape, validTestData.shape))
    return (testImgId, validTestData)

## input
## (testImgId, predictionResult)
## ([sample x 1], [sample x 1])
##  output
## result
## [sample x 2]
def concatenateResult(testImgId, predictionResult):
    result = np.concatenate((testImgId, predictionResult), axis=1)
    result = result.astype(int)
    return result

## input
## (result, method='', outputFileName='')
## ([sample x 2], ...)
def saveResult(result, method='', outputFileName=''):
    header = parameters.resultHeader
    timeSuffix = datetime.datetime.now().strftime("%b%d%H%M")
    if outputFileName == '':
        if method != '':
            outputFileName += method + '_'
        outputFileName += timeSuffix
    if not outputFileName.endswith('.csv'):
        outputFileName += '.csv'
    outputFilePath = parameters.resultFolder + outputFileName
    print("Saving result to : '{}'...".format(outputFilePath))
    f = open(outputFilePath, 'w', newline='')
    csvWriter = csv.writer(f)
    csvWriter.writerow(header)
    for row in result:
        csvWriter.writerow(row)
    f.close()
    print("Save complete!")