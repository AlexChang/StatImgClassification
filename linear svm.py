import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import csv


import utils

def main():
    (trainInput, trainTarget) = utils.getTrainData()
    (testImgId, validTestData) = utils.getTestData()

    lin_clf = svm.LinearSVC(C=1, multi_class='ovr', verbose=True, penalty='l1', dual=False)
    print('Training...')
    lin_clf.fit(trainInput, trainTarget.ravel())
    print('Training complete!')

    print('Predicting...')
    predictionResult = lin_clf.predict(validTestData)
    predictionResult = np.expand_dims(predictionResult, axis=1)
    print('Prediction complete!')

    result = utils.concatenateResult(testImgId, predictionResult)

    utils.saveResult(result, 'lin_svm_ovr_penl1_dualF')

if __name__ == '__main__':
    main()
