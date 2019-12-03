from sklearn.base import ClassifierMixin
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression
import itertools as it
import numpy as np
class PairwiseCoupling_LR(BaseEstimator,ClassifierMixin):
    def __init__(self, classes=3, seed=42, penalty='l2', C=1.0):
        self.nclasses=classes
        self.seed=seed
        self.baselearner_class=LogisticRegression
        self.kwargs={'penalty': penalty, 'C':C}
        self.params = {'classes': classes, 'seed': seed, 'penalty': penalty, 'C':C}
    
    def fit(self, X, y=None):
        self.classes_=np.unique(y)
        self.pairClassifier = []
        X_byClasses = []
        for _ in range(self.nclasses):
            X_byClasses.append([])
        for inst_index in range(y.size):
            X_byClasses[y[inst_index]].append(X[inst_index])
        for pair in it.combinations(range(self.nclasses),2):
            X_pair=np.vstack((X_byClasses[pair[0]], X_byClasses[pair[1]]))
            y_pair=np.hstack((np.full(len(X_byClasses[pair[0]]),0),np.full(len(X_byClasses[pair[1]]),1)))
            baselearner_pair=self.baselearner_class(**self.kwargs)
            baselearner_pair.fit(X_pair,y_pair)
            self.pairClassifier.append(baselearner_pair)
            
    def predict_proba(self, X):
        result = []
        for i in range(len(X)):
            single_X = X[i]
            invertedSumByClass = []
            # -(n-2) part
            for _ in range(self.nclasses):
                invertedSumByClass.append(2-self.nclasses)
            #print(invertedSumByClass)
            p = 0
            # 1/p_ij for every class
            for pair in it.combinations(range(self.nclasses),2):
                proba_pair=self.pairClassifier[p].predict_proba([single_X])
                p += 1
                #if (p==1):
                #    print(proba_pair)
                invertedSumByClass[pair[0]] += 1.0/proba_pair[0][0]
                invertedSumByClass[pair[1]] += 1.0/proba_pair[0][1] 
            sum = 0
            for k in range(self.nclasses):
                invertedSumByClass[k] = 1.0/invertedSumByClass[k]
                sum += invertedSumByClass[k]
            # normalise
            for k in range(self.nclasses):
                invertedSumByClass[k] = invertedSumByClass[k]/sum
                if (sum == 0):
                    invertedSumByClass[k] = 0
            result.append(invertedSumByClass)
        return np.asarray(result)
    
    def get_params(self, deep=False):
        return self.params
    
    def set_params(self, **params):
        self.kwargs = params;
        self.params['penalty'] = params['penalty']
        self.params['C'] = params['C']
        return self