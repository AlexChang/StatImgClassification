import numpy as np
import datetime
import argparse


from sklearn.decomposition import PCA


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
    parser.add_argument('--pca', action='store_true', default=False, help='pca')
    parser.add_argument('--dim', type=int, default=1289, help='pca dim')
    parser.add_argument('--best', action='store_true', default=False, help='load best model/params')
    parser.add_argument('--tm', action='store_true', default=False, help='test model')
    parser.add_argument('--tp', action='store_true', default=False, help='test parameters')
    parser.add_argument('--fn', type=str, default='', help='file name')
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


def fitPCA(trainInput, dim=1289):
    #pca = PCA(svd_solver='full', n_components='mle')
    pca = PCA(n_components=dim)
    print("PCA fitting with parameters: {}".format(pca.get_params()))
    pca.fit(trainInput)
    print('Fit complete!')
    return pca


def classify(args, method):
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%b%d%H%M")

    # read training & test data
    (trainInput, trainTarget) = utils.loadTrainData()
    (testImgId, validTestData) = utils.loadTestData()

    # pca
    if args.pca:
        pca = fitPCA(trainInput, args.dim)
        trainInput = pca.transform(trainInput)
        validTestData = pca.transform(validTestData)

    # choose method
    parameter = Parameter(method)

    # get classifier
    if method == 'svm':
        clf = svm.getModel()
    elif method == 'lin_svm':
        #parameter.addParameter('multi_class', 'ovr')
        #parameter.addParameter('loss', 'l2')
        #parameter.addParameter('penalty', 'l1')
        #parameter.addParameter('dual', False)
        #parameter.addParameter('C', 10)
        #parameter.addParameter('max_iter', 10000)
        clf = linear_svm.getModel()
    elif method == 'knn':
        clf = knn.getModel()
    elif method == 'lda':
        clf = lda.getModel()
    elif method == 'qda':
        clf = qda.getModel()
    elif method == 'rf':
        clf = rf.getModel()
    elif method == 'ada':
        clf = adaboost.getModel()
    elif method == 'lr':
        clf = lr.getModel()
    elif method == 'rc':
        clf = rc.getModel()
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


def testModel(args, method, modelFileName='', isCV=True):
    if args.best:
        clf = utils.loadBestModel(method, isCV=isCV)
    else:
        clf = utils.loadModel(modelFileName)
    print(clf)


def testParameter(args, method, parameterFileName='', isCV=True):
    if args.best:
        parameterDict = utils.loadBestParameters(method, isCV=isCV)
    else:
        parameterDict = utils.loadParameters(parameterFileName)
    print(parameterDict)


def main():
    args = initArgParser()
    method = args.mode.lower()
    if args.tm:
        testModel(args, method)
    if args.tp:
        testParameter(args, method)
    if not (args.tm or args.tp):
        if method == 'all':
            for method in supportedMethods:
                classify(args, method)
                print()
        elif not method in supportedMethods:
            raise ValueError("unsupported classification method: {}".format(method))
        else:
            classify(args, method)

if __name__ == '__main__':
    main()
