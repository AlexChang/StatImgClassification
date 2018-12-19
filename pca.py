import utils

import matplotlib.pyplot as plt
import numpy as np

from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

def compute_scores(X):
    pca = PCA(svd_solver='full')
    fa = FactorAnalysis()

    pca_scores, fa_scores = [], []
    for n in n_components:
        print("Processing N={}".format(n))
        pca.n_components = n
        fa.n_components = n
        pca_scores.append(np.mean(cross_val_score(pca, X, cv=5)))
        fa_scores.append(np.mean(cross_val_score(fa, X, cv=5)))

    return pca_scores, fa_scores

(trainInput, trainTarget) = utils.loadTrainData()
#(testImgId, validTestData) = utils.loadTestData()

numStart = 100
numEnd = 185
numSpan = 5

n_components = np.arange(numStart, numEnd, numSpan)  # options for n_components

X = trainInput
title = 'Noise'
pca_scores, fa_scores = compute_scores(X)
n_components_pca = n_components[np.argmax(pca_scores)]
n_components_fa = n_components[np.argmax(fa_scores)]

#pca = PCA(svd_solver='full', n_components='mle')
#pca.fit(X)
#n_components_pca_mle = pca.n_components_

print("best n_components by PCA CV = %d" % n_components_pca)
print("best n_components by FactorAnalysis CV = %d" % n_components_fa)
#print("best n_components by PCA MLE = %d" % n_components_pca_mle)

plt.figure()
plt.plot(n_components, pca_scores, 'b', label='PCA scores')
plt.plot(n_components, fa_scores, 'r', label='FA scores')
plt.axvline(n_components_pca, color='b', label='PCA CV: %d' % n_components_pca, linestyle='--')
plt.axvline(n_components_fa, color='r', label='FactorAnalysis CV: %d' % n_components_fa, linestyle='--')
#plt.axvline(n_components_pca_mle, color='k',
#            label='PCA MLE: %d' % n_components_pca_mle, linestyle='--')

plt.xlabel('nb of components')
plt.ylabel('CV scores')
plt.legend(loc='lower right')
plt.title(title)

figName = "pca_start={}_end={}_span={}.png".format(numStart, numEnd, numSpan)
plt.savefig(utils.graphFolder + figName)
plt.close()