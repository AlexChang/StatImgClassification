import utils
import csv

def loadResultData(resultFileName):
    resultFileFullName = utils.resultFolder + resultFileName
    print("Loading result data from : '{}'...".format(resultFileFullName))
    f = open(resultFileFullName)
    result = list(csv.reader(f))[1:]
    f.close()
    print('Done!')
    return result

def loadResultDataList(resultNameList):
    resultDataList = []
    for resultFileName in resultNameList:
        resultDataList.append(loadResultData(resultFileName))
    return resultDataList

def compareResultData(resultDataList):
    sameResultData = []
    for i in range(len(resultDataList[0])):
        curRowResultData = resultDataList[0][i]
        flag = True
        for j in range(len(resultDataList)):
            if resultDataList[j][i] != curRowResultData:
                flag = False
                break
        if flag:
            sameResultData.append(curRowResultData)
    return sameResultData

def loadRawTestData():
    print("Loading test data from : '{}'...".format(utils.testDataPath))
    f = open(utils.testDataPath)
    result = list(csv.reader(f))[1:]
    f.close()
    print('Done!')
    return result

def loadRawTrainData():
    print("Loading train data from : '{}'...".format(utils.trainDataPath))
    f = open(utils.trainDataPath)
    result = list(csv.reader(f))
    f.close()
    print('Done!')
    return result

def convertTestToTrain(testData, sameResultData):
    sameIdx = 0
    newTrainData = []
    for i in range(len(testData)):
        if (testData[i][0] == sameResultData[sameIdx][0]):
            testData[i].append(sameResultData[sameIdx][1])
            newTrainData.append(testData[i])
            sameIdx += 1
        if sameIdx == len(sameResultData):
            break
    return newTrainData

def concatenateTrainData(trainData, newTrainData):
    trainData.extend(newTrainData)

def saveTainData(trainData, fileName):
    fullFileName = utils.dataFolder + fileName
    if not fullFileName.endswith('.csv'):
        fullFileName += '.csv'
    print("Saving new train data to : '{}'...".format(fullFileName))
    f = open(fullFileName, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(trainData)
    f.close()
    print('Done!')

def augmentTrainData(outputFileName):
    resultNameList = ['linsvm_CV_accuracy=0.98692_pre=minmax_C=0.01_Dec211019.csv',
                      'rigclf_CV_accuracy=0.98782_pre=l2_Dec211227.csv',
                      'logreg_CV_accuracy=0.98718_max_iter=10000_class_weight=None_solver=liblinear_C=1_Dec201428.csv']

    resultDataList = loadResultDataList(resultNameList)
    sameResultData = compareResultData(resultDataList)

    testData = loadRawTestData()
    trainData = loadRawTrainData()
    newTrainData = convertTestToTrain(testData, sameResultData)
    concatenateTrainData(trainData, newTrainData)
    saveTainData(trainData, outputFileName)

def iterPurify(fileName, outputFileName):
    resultData = loadResultData(fileName)
    trainData = loadRawTrainData()
    newTrainData = []
    diffCount = 0
    for i in range(len(trainData)):
        if i <= 7800:
            newTrainData.append(trainData[i])
        else:
            resultIdx = int(trainData[i][0])
            if trainData[i][-1] != resultData[resultIdx][-1]:
                diffCount += 1
            else:
                newTrainData.append(trainData[i])
    print('Diff Count: {}'.format(diffCount))
    saveTainData(newTrainData, outputFileName)

def iterModify(fileName, outputFileName):
    resultData = loadResultData(fileName)
    trainData = loadRawTrainData()
    newTrainData = []
    diffCount = 0
    for i in range(len(trainData)):
        if i <= 7800:
            newTrainData.append(trainData[i])
        else:
            resultIdx = int(trainData[i][0])
            if trainData[i][-1] != resultData[resultIdx][-1]:
                trainData[i][-1] = resultData[resultIdx][-1]
                diffCount += 1
            newTrainData.append(trainData[i])
    print('Diff Count: {}'.format(diffCount))
    saveTainData(newTrainData, outputFileName)

def main():
    #augmentTrainData('newData')
    #iterPurify('rigclf_CV_accuracy=0.98495_pre=l2_Jan141637.csv', 'newData2')
    #iterPurify('rigclf_CV_accuracy=0.98767_pre=l2_Jan151340.csv', 'newData3')
    #iterPurify('rigclf_CV_accuracy=0.98793_pre=l2_Jan151400.csv', 'newData4')
    iterPurify('rigclf_CV_accuracy=0.98775_pre=l2_Jan151406.csv', 'newData5')
    #iterModify('rigclf_CV_accuracy=0.98495_pre=l2_Jan141637.csv', 'newModData2')

if __name__ == '__main__':
    main()