import numpy as np
import datetime
import argparse


from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV


import model.svm as svm
import model.linear_svm as linear_svm
import model.knn as knn
import model.lda as lda
import model.qda as qda
import model.rf as rf
import model.adaboost as adaboost
import utils

supportedMethods = ['svm', 'lin_svm', 'knn', 'lda', 'qda', 'rf', 'ada']

def initArgParser():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--mode', type=str, default='knn')
    parser.add_argument('--best', action='store_true', default=False, help='')
    parser.add_argument('--tm', action='store_true', default=False, help='test model')
    parser.add_argument('--tp', action='store_true', default=False, help='test parameters')
    parser.add_argument('--sd', action='store_true', default=False, help='sample data set')
    parser.add_argument('--job', type=int, default=2)
    args = parser.parse_args()
    return args

def predict(clf, validTestData):
    print('Predicting...')
    predictionResult = clf.predict(validTestData)
    predictionResult = np.expand_dims(predictionResult, axis=1)
    print('Prediction complete!')
    return predictionResult

def cv(args, method, isPredict=True):
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
    else:
        raise ValueError("unsupported classification method: {}".format(method))

    # set cv and scores
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)
    #scores = ['accuracy', 'precision_macro', 'precision_macro_micro']
    scores = ['accuracy']

    # cv
    for score in scores:
        print("# Tuning hyper-parameters for {}".format(score))
        print()

        clf = GridSearchCV(clf, param_grid, cv=cv, scoring=score, n_jobs=args.job)
        clf.fit(trainInput, trainTarget)

        scoreResult = utils.getScoreResult(clf)
        print(scoreResult)

        # save best parameter
        bestParameterFileName = utils.generateOutputFileName('json', method=method, parameters='cv_score={}'.format(score), timestamp=timestamp, isBest=True)
        utils.saveParameters(clf.best_params_, bestParameterFileName)

        # save cv grid scores
        scoreFileName = utils.generateOutputFileName('txt', method=method, parameters='cv_score={}'.format(score), timestamp=timestamp)
        utils.saveScores(scoreResult, scoreFileName)

        # save best model
        bestModelFileName = utils.generateOutputFileName('joblib', method=method, parameters='cv_score={}'.format(score), timestamp=timestamp, isBest=True)
        utils.saveModel(clf, bestModelFileName)

        if isPredict:
            if args.sd:
                print("Ignore prediction operation since using sample data set: iris! ")
            else:
                # read test data
                (testImgId, validTestData) = utils.loadTestData()
                predictionResult = predict(clf, validTestData)

                # save result
                result = utils.concatenateResult(testImgId, predictionResult)
                resultFileName = utils.generateOutputFileName('csv', method=method, parameters='cv_score={}'.format(score), timestamp=timestamp, isBest=True)
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
