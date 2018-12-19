import numpy as np
import datetime
import argparse


from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import linear_model

import model.svm as svm
import model.linear_svm as linear_svm
import model.knn as knn
import model.lda as lda
import model.qda as qda
import model.rf as rf
import model.adaboost as adaboost
import model.lr as lr
import model.rc as rc
import utils

#supportedMethods = ['svm', 'lin_svm', 'knn', 'lda', 'qda', 'rf', 'ada', 'lr', 'rc']
supportedMethods = ['lin_svm', 'knn', 'lda', 'lr', 'rc']

def initArgParser():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--mode', type=str, default='linReg')
    parser.add_argument('--pca', action='store_true', default=False, help='pca')
    parser.add_argument('--dim', type=int, default=1289, help='pca dim')
    parser.add_argument('--gs', action='store_true', default=False, help='grid search')
    parser.add_argument('--predict', action='store_false', default=False, help='NOT predict on test set')
    parser.add_argument('--best', action='store_true', default=False, help='load best model/params')
    parser.add_argument('--tm', action='store_true', default=False, help='test model')
    parser.add_argument('--tp', action='store_true', default=False, help='test parameters')
    parser.add_argument('--sd', action='store_true', default=False, help='sample data set')
    parser.add_argument('--sm', action='store_false', default=False, help='NOT store model')
    parser.add_argument('--job', type=int, default=4)
    parser.add_argument('--fn', type=str, default='', help='file name')
    args = parser.parse_args()
    return args


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


def cv(args, method):
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%b%d%H%M")

    # read training data
    if args.sd:
        # use sample data set
        print("Using sample data set: iris!")
        iris = datasets.load_iris()
        (trainInput, trainTarget) = iris.data, iris.target
        print("Shape of training data: input: {}, target: {}".format(trainInput.shape, trainTarget.shape))
    else:
        (trainInput, trainTarget) = utils.loadTrainData()

    # pca
    if args.pca:
        pca = fitPCA(trainInput, args.dim)
        trainInput = pca.transform(trainInput)

    # get classifier and param_grid
    if method == 'svm':
        param_grid = svm.param_grid
        clf = svm.getModel()
    elif method == 'lin_svm':
        param_grid = linear_svm.param_grid
        clf = linear_svm.getModel()
    elif method == 'knn':
        param_grid = knn.param_grid
        clf = knn.getModel()
    elif method == 'lda':
        param_grid = lda.param_grid
        clf = lda.getModel()
    elif method == 'qda':
        param_grid = qda.param_grid
        clf = qda.getModel()
    elif method == 'rf':
        param_grid = rf.param_grid
        clf = rf.getModel()
    elif method == 'ada':
        param_grid = adaboost.param_grid
        clf = adaboost.getModel()
    elif method == 'lr':
        param_grid = lr.param_grid
        clf = lr.getModel()
    elif method == 'rc':
        param_grid = rc.param_grid
        clf = rc.getModel()
    else:
        clf = linear_model.LinearRegression()

    # set cv and scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)
    #scores = ['accuracy', 'precision_macro', 'precision_macro_micro']
    scores = ['accuracy']

    # cv
    for score in scores:
        if args.gs:
            print("# Tuning hyper-parameters for {}".format(score))
            clf = GridSearchCV(clf, param_grid, cv=cv, scoring=score, n_jobs=args.job)
            clf.fit(trainInput, trainTarget)
            scoreResult = utils.getGridSearchScoreResult(clf)
            print(scoreResult)
            parameter = 'CVGridSearch_score={}'.format(score)
            # save best parameter
            bestParameterFileName = utils.generateOutputFileName('json', method=method, parameters=parameter,
                                                                 timestamp=timestamp, isBest=True)
            utils.saveParameters(clf.best_params_, bestParameterFileName)
            # save cv grid scores
            scoreFileName = utils.generateOutputFileName('txt', method=method, parameters=parameter,
                                                         timestamp=timestamp)
            utils.saveScores(scoreResult, scoreFileName)
            # save best model
            if args.sm:
                bestModelFileName = utils.generateOutputFileName('joblib', method=method, parameters=parameter,
                                                                 timestamp=timestamp, isBest=True)
                utils.saveModel(clf, bestModelFileName)
        else:
            print("# Cross Validation for {}".format(score))
            cvs = cross_val_score(clf, trainInput, trainTarget, cv=cv, scoring=score, n_jobs=args.job)
            scoreResult = utils.getCVScoreResult(cvs, clf.get_params())
            print(scoreResult)
            parameter = 'CV_score={}'.format(score)
            # save parameter
            parameterFileName = utils.generateOutputFileName('json', method=method, parameters=parameter,
                                                                 timestamp=timestamp, isBest=False)
            utils.saveParameters(clf.get_params(), parameterFileName)
            # save cv grid scores
            scoreFileName = utils.generateOutputFileName('txt', method=method, parameters=parameter,
                                                         timestamp=timestamp)
            utils.saveScores(scoreResult, scoreFileName)
            # save model
            if args.sm:
                modelFileName = utils.generateOutputFileName('joblib', method=method, parameters=parameter,
                                                                 timestamp=timestamp, isBest=True)
                utils.saveModel(clf, modelFileName)

        if args.predict:
            if args.sd:
                print("Ignore prediction operation since using sample data set: iris! ")
            else:
                # read test data
                (testImgId, validTestData) = utils.loadTestData()
                # pca
                if args.pca:
                    validTestData = pca.transform(validTestData)
                # predict
                predictionResult = predict(clf, validTestData)

                # save result
                result = utils.concatenateResult(testImgId, predictionResult)
                if args.gd:
                    resultFileName = utils.generateOutputFileName('csv', method=method, parameters=parameter,
                                                                  timestamp=timestamp, isBest=True)
                else:
                    resultFileName = utils.generateOutputFileName('csv', method=method, parameters=parameter,
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
        cv(args, method)


if __name__ == '__main__':
    main()
