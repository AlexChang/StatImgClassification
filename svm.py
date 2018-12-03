import numpy as np
from sklearn import svm
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import cross_val_score
import csv


import utils

def main():
    (trainInput, trainTarget) = utils.getTrainData()
    (testImgId, validTestData) = utils.getTestData()

    clf = svm.SVC(C=1, decision_function_shape='ovo', class_weight='balanced', verbose=True)
    print('Training...')
    clf.fit(trainInput, trainTarget.ravel())
    print('Training complete!')

    print('Predicting...')
    predictionResult = clf.predict(validTestData)
    predictionResult = np.expand_dims(predictionResult, axis=1)
    print('Prediction complete!')

    result = utils.concatenateResult(testImgId, predictionResult)

    utils.saveResult(result, 'svm_ovo_b')

if __name__ == '__main__':
    main()

'''
### train data
rawData = np.loadtxt('data/train.csv', skiprows=1, delimiter=',')
validData = rawData[:, 1:]
data, target = np.split(validData, (validData.shape[1] - 1, ), axis=1)
print("data: {}, target: {}".format(data.shape, target.shape))

### model
clf = svm.SVC(kernel='linear', C=1.1)

### cv
cv = ShuffleSplit(n_splits=3, test_size=0.3)
scores = cross_val_score(clf, data, target.ravel(), cv=cv)
print(scores)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

### fit
clf.fit(data, target.ravel())

### test data
testRawData = np.loadtxt('data/test.csv', skiprows=1, delimiter=',')
imgId = testRawData[:, :1]
validTestData = testRawData[:, 1:]
print("test data: {}".format(validTestData.shape))

### predict
predictResult = clf.predict(validTestData)
predictResult = np.expand_dims(predictResult, axis=1)

### output result
result = np.concatenate((imgId, predictResult), axis=1)
result = result.astype(int)
#np.savetxt('result.csv', result, delimiter=',')
header = ['id', 'categories']
out = open('resutl2.csv', 'w', newline='')
csvWriter = csv.writer(out)
csvWriter.writerow(header)
for item in result:
    csvWriter.writerow(item)
out.close()
'''