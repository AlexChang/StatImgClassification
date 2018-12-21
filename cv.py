import numpy as np
import datetime
import argparse


from sklearn import datasets
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import PCA
from sklearn import preprocessing


import model.svm as svm
import model.linSvm as linSvm
import model.knn as knn
import model.lda as lda
import model.qda as qda
import model.rf as rf
import model.adaBst as ada_bst
import model.logReg as logReg
import model.rigClf as rigClf
from hyperParameter import HyperParameter
import utils

#supportedMethods = ['svm', 'linSvm', 'knn', 'lda', 'qda', 'rf', 'ada_bst', 'logReg', 'rigClf']
supportedMethods = ['linSvm', 'knn', 'lda', 'logReg', 'rigClf']

def initArgParser():
    parser = argparse.ArgumentParser(description='Image Classifier')
    parser.add_argument('--mode', type=str, default='lda')
    parser.add_argument('--pca', action='store_false', default=True, help='NOT pca')
    parser.add_argument('--dim', type=int, default=130, help='pca dim')
    parser.add_argument('--pre', type=str, default='minmax')
    parser.add_argument('--gs', action='store_true', default=True, help='grid search')
    parser.add_argument('--predict', action='store_false', default=True, help='NOT predict on test set')
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


def fitPCA(trainInput, dim=130):
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

    # preprocessing
    if args.pre == 'scale':
        trainInput = preprocessing.scale(trainInput)
    elif args.pre == 'l2':
        trainInput = preprocessing.normalize(trainInput, norm='l2')
    elif args.pre == 'l1':
        trainInput = preprocessing.normalize(trainInput, norm='l1')
    elif args.pre == 'minmax':
        min_max_scaler = preprocessing.MinMaxScaler()
        trainInput = min_max_scaler.fit_transform(trainInput)

    # pca
    if args.pca:
        pca = fitPCA(trainInput, args.dim)
        trainInput = pca.transform(trainInput)

    # hyper params
    hyperParameter = HyperParameter(method)
    if not args.gs:
        # lda
        #hyperParameter.addParameter('C', 10)
        #hyperParameter.addParameter('n_components', 130)
        prior = [1/12 for x in range(12)]
        prior = np.asarray(prior)
        #hyperParameter.addParameter('priors', prior)
        #hyperParameter.addParameter('solver', 'eigen')

        ### log Reg Pca
        #hyperParameter.addParametersByDict({'max_iter': 10000, 'tol': 0.01, 'C': 100, 'class_weight': 'balanced', 'dual': False, 'multi_class': 'multinomial', 'solver': 'saga', 'penalty': 'l2'})
        #hyperParameter.addParametersByDict({'solver': 'liblinear', 'tol': 0.01, 'penalty': 'l1', 'max_iter': 10000, 'multi_class': 'ovr', 'C': 1, 'class_weight': None})
        #hyperParameter.addParametersByDict({'penalty': 'l1', 'max_iter': 10000, 'class_weight': 'balanced', 'solver': 'saga', 'dual': False, 'multi_class': 'multinomial', 'tol': 0.01, 'C': 10})
        #hyperParameter.addParametersByDict({'max_iter': 10000, 'class_weight': None, 'solver': 'liblinear', 'C': 1, 'penalty': 'l2', 'multi_class': 'ovr', 'dual': True, 'tol': 0.01})
        #hyperParameter.addParametersByDict({'penalty': 'l2', 'dual': False, 'class_weight': None, 'tol': 0.01, 'C': 0.1, 'solver': 'saga', 'multi_class': 'multinomial', 'max_iter': 10000})
        #hyperParameter.addParametersByDict({'solver': 'saga', 'C': 0.1, 'multi_class': 'multinomial', 'dual': False, 'penalty': 'l1', 'class_weight': None, 'max_iter': 10000, 'tol': 0.01})

        ### lin Svm
        #hyperParameter.addParametersByDict({'penalty': 'l2', 'dual': True, 'C': 0.01, 'loss': 'squared_hinge', 'tol': 0.01, 'max_iter': 10000})
        #hyperParameter.addParametersByDict({'tol': 0.01, 'max_iter': 10000, 'penalty': 'l2', 'C': 0.01, 'loss': 'squared_hinge', 'dual': False})
        #hyperParameter.addParameter('C', 0.01)

    # get classifier and param_grid
    if method == 'svm':
        param_grid = svm.param_grid
        clf = svm.getModel(hyperParameter)
    elif method == 'linsvm':
        param_grid = linSvm.param_grid
        clf = linSvm.getModel(hyperParameter)
    elif method == 'knn':
        param_grid = knn.param_grid
        clf = knn.getModel(hyperParameter)
    elif method == 'lda':
        param_grid = lda.param_grid
        clf = lda.getModel(hyperParameter)
    elif method == 'qda':
        param_grid = qda.param_grid
        clf = qda.getModel(hyperParameter)
    elif method == 'rf':
        param_grid = rf.param_grid
        clf = rf.getModel(hyperParameter)
    elif method == 'adabst':
        param_grid = ada_bst.param_grid
        clf = ada_bst.getModel(hyperParameter)
    elif method == 'logreg':
        param_grid = logReg.param_grid
        clf = logReg.getModel(hyperParameter)
    elif method == 'rigclf':
        param_grid = rigClf.param_grid
        clf = rigClf.getModel(hyperParameter)
    else:
        raise ValueError("unsupported classification method: {}".format(method))

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
            parameter = 'CVGridSearch_{}={:.5f}'.format(score, clf.best_score_)
            if not args.pre == '':
                parameter += '_pre={}'.format(args.pre)
            if args.pca:
                parameter += '_pca={}'.format(args.dim)
            parameter += '_{}'.format(utils.parameterDictToString(clf.best_params_))
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
            clf.fit(trainInput, trainTarget)
            scoreResult = utils.getCVScoreResult(cvs, clf.get_params())
            print(scoreResult)
            parameter = 'CV_{}={:.5f}'.format(score, cvs.mean())
            if not args.pre == '':
                parameter += '_pre={}'.format(args.pre)
            if args.pca:
                parameter += '_pca={}'.format(args.dim)
            if not hyperParameter.parameterDict == {}:
                parameter += '_{}'.format(hyperParameter.toString())
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
                # preprocessing
                if args.pre == 'scale':
                    validTestData = preprocessing.scale(validTestData)
                elif args.pre == 'l2':
                    validTestData = preprocessing.normalize(validTestData, norm='l2')
                elif args.pre == 'l1':
                    validTestData = preprocessing.normalize(validTestData, norm='l1')
                elif args.pre == 'minmax':
                    validTestData = min_max_scaler.transform(validTestData)
                # pca
                if args.pca:
                    validTestData = pca.transform(validTestData)
                # predict
                predictionResult = predict(clf, validTestData)

                # save result
                result = utils.concatenateResult(testImgId, predictionResult)
                if args.gs:
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
