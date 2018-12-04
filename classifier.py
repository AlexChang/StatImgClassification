import numpy as np

import svm
import linear_svm
import knn
import utils

def train(clf, trainInput, trainTarget):
    print('Training...')
    clf.fit(trainInput, trainTarget)
    print('Training complete!')
    return clf

def predict(clf, validTestData):
    print('Predicting...')
    predictionResult = clf.predict(validTestData)
    predictionResult = np.expand_dims(predictionResult, axis=1)
    print('Prediction complete!')
    return predictionResult

def classify(method):
    (trainInput, trainTarget) = utils.getTrainData()
    (testImgId, validTestData) = utils.getTestData()

    method = method.lower()
    if method == 'knn':
        parameter = knn.KNNParameter()
        parameter.addParameter('weights', 'uniform')
        clf = knn.getModel(parameter.parameterDict)
    elif method == 'svm':
        parameter = svm.SVMParameter()
        parameter.addParameter('multi_class', 'ovr')
        clf = svm.getModel(parameter.parameterDict)
    elif method == 'linear_svm':
        parameter = linear_svm.LINEARSVMParameter()
        clf = linear_svm.getModel(parameter.parameterDict)
    else:
        raise ValueError("unrecognized classification method: '%s'" % method)

    clf = train(clf, trainInput, trainTarget)
    predictionResult = predict(clf, validTestData)

    result = utils.concatenateResult(testImgId, predictionResult)
    utils.saveResult(result, method, parameter.toString())

def main():
    classify('linear_svm')

if __name__ == '__main__':
    main()
