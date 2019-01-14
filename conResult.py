import utils
import csv

resultNameList = ['linsvm_CV_accuracy=0.98692_pre=minmax_C=0.01_Dec211019.csv', 'rigclf_CV_accuracy=0.98782_pre=l2_Dec211227.csv', 'logreg_CV_accuracy=0.98718_max_iter=10000_class_weight=None_solver=liblinear_C=1_Dec201428.csv']

def loadResultData(resultFileName):
    resultFileFullName = utils.resultFolder + resultFileName
    f = open(resultFileFullName)
    result = list(csv.reader(f))[1:]
    f.close()
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

resultDataList = loadResultDataList(resultNameList)
sameResultData = compareResultData(resultDataList)

def loadRawTestData():
    print('Reading test data...')
    f = open(utils.testDataPath)
    result = list(csv.reader(f))[1:]
    f.close()
    return result

def loadRawTrainData():
    print('Reading train data...')
    f = open(utils.trainDataPath)
    result = list(csv.reader(f))
    f.close()
    return result

def modTestData(testData, sameResultData):
    sameIdx = 0
    newData = []
    for i in range(len(testData)):
        if (testData[i][0] == sameResultData[sameIdx][0]):
            testData[i].append(sameResultData[sameIdx][1])
            newData.append(testData[i])
            sameIdx += 1
        if sameIdx == len(sameResultData):
            break
    return newData

def genTrainData(trainData, newData):
    trainData.extend(newData)

def saveTainData(trainData):
    fileName = utils.dataFolder + 'newData.csv'
    print('Saving new train data...')
    #print(trainData)
    f = open(fileName, 'w', newline='')
    writer = csv.writer(f)
    writer.writerows(trainData)
    f.close()


testData = loadRawTestData()
trainData = loadRawTrainData()
newData = modTestData(testData, sameResultData)
genTrainData(trainData, newData)
saveTainData(trainData)
