import numpy as np
import datetime
import argparse
from sklearn.decomposition import PCA, FactorAnalysis

import model.svm as svm
import model.linear_svm as linear_svm
import model.knn as knn
import model.lda as lda
import model.qda as qda
import model.rf as rf
import model.adaboost as adaboost
import model.lr as lr
import model.rc as rc
from parameter import Parameter
import utils

supportedMethods = ['svm', 'lin_svm', 'knn', 'lda', 'qda', 'rf', 'ada', 'lr', 'rc']

def initArgParser():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--mode', type=str, default='lr')
    parser.add_argument('--bp', action='store_true', default=False)
    parser.add_argument('--test', action='store_true', default=False)
    args = parser.parse_args()
    return args

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

def classify(args, method):
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%b%d%H%M")

    # read training & test data
    (trainInput, trainTarget) = utils.loadTrainData()
    (testImgId, validTestData) = utils.loadTestData()

    pca = PCA(svd_solver='full', n_components='mle')
    pca.fit(trainInput)
    trainInput = pca.transform(trainInput)
    validTestData = pca.transform(validTestData)

    # choose method
    method = method.lower()
    parameter = Parameter(method)
    if args.bp:
        parameter.loadBestParameter()

    # get classifier
    if method == 'svm':
        if not args.bp:
            #parameterDict = {'kernel': 'linear', 'decision_function_shape': 'ovr'}
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = svm.getModel(parameter.parameterDict)
    elif method == 'lin_svm':
        if not args.bp:
            #parameter.addParameter('multi_class', 'ovr')
            #parameter.addParameter('loss', 'l2')
            #parameter.addParameter('penalty', 'l1')
            #parameter.addParameter('dual', False)
            #parameter.addParameter('C', 10)
            #parameter.addParameter('max_iter', 10000)
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = linear_svm.getModel(parameter.parameterDict)
    elif method == 'knn':
        if not args.bp:
            #parameter.addParameter('n_neighbors', 15)
            #parameter.addParameter('weights', 'distance')
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = knn.getModel(parameter.parameterDict)
    elif method == 'lda':
        if not args.bp:
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = lda.getModel(parameter.parameterDict)
    elif method == 'qda':
        if not args.bp:
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = qda.getModel(parameter.parameterDict)
    elif method == 'rf':
        if not args.bp:
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = rf.getModel(parameter.parameterDict)
    elif method == 'ada':
        if not args.bp:
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = adaboost.getModel(parameter.parameterDict)
    elif method == 'lr':
        if not args.bp:
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = lr.getModel(parameter.parameterDict)
    elif method == 'rc':
        if not args.bp:
            parameterDict = {}
            parameter.addParametersByDict(parameterDict)
        clf = rc.getModel(parameter.parameterDict)
    else:
        raise ValueError("unsupported classification method: {}".format(method))

    # save parameters
    parameterFileName = utils.generateOutputFileName('json', method=method, parameters=parameter.toString(),
                                                     timestamp=timestamp)
    utils.saveParameters(clf.get_params(), outputFileName=parameterFileName)

    # train & predict
    clf = train(clf, trainInput, trainTarget)
    predictionResult = predict(clf, validTestData)

    # save result
    result = utils.concatenateResult(testImgId, predictionResult)
    resultFileName = utils.generateOutputFileName('csv', method=method, parameters=parameter.toString(),
                                                  timestamp=timestamp)
    utils.saveResult(result, outputFileName=resultFileName)


def main():
    args = initArgParser()
    method = args.mode.lower()
    if not args.test:
        if method == 'all':
            for method in supportedMethods:
                classify(args, method)
                print()
        elif not method in supportedMethods:
            raise ValueError("unsupported classification method: {}".format(method))
        else:
            classify(args, method)
    else:
        utils.loadBestParameters('lin_svm')

if __name__ == '__main__':
    main()
