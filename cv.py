import numpy as np
import datetime
import argparse
import glob
from sklearn.model_selection import cross_val_score

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report


import svm
import linear_svm
import knn
import lda
import qda
from parameter import Parameter
import utils

supportedMethods = ['svm', 'lin_svm', 'knn', 'lda', 'qda', 'rf', 'ada']

def initArgParser():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--mode', type=str, default='qda')
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

def cv(args, method):
    # get timestamp
    timestamp = datetime.datetime.now().strftime("%b%d%H%M")

    # read train
    (trainInput, trainTarget) = utils.getTrainData()

    tuned_parameters = [{'penalty': ['l1', 'l2'], 'loss': ['hinge', 'squared_hinge'],
                         'C': [1, 10, 100, 1000], 'tol': [1e-3, 1e-4, 1e-5], 'max_iter': 5000}]

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=9)
    scores = cross_val_score(clf, trainInput, trainTarget, cv=cv)
    print(scores)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    score = 'precision'
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(), tuned_parameters, cv=5,
                       scoring='%s_macro' % score)
    clf.fit(X_train, y_train)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

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
            # parameter.addParameter('multi_class', 'ovr')
            # parameter.addParameter('loss', 'l2')
            parameter.addParameter('penalty', 'l1')
            parameter.addParameter('dual', False)
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
    else:
        raise ValueError("unsupported classification method: {}".format(method))

    # save parameters
    parameterFileName = utils.generateOutputFileName('json', method=method, parameters=parameter.toString(),
                                                     timestamp=timestamp)
    utils.saveParameters(clf, outputFileName=parameterFileName)

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
        utils.getBestParameters('lin_svm')

if __name__ == '__main__':
    main()
