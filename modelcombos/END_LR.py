import NestedDichotomies.nd as nd
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.linear_model import LogisticRegression
import random
import numpy as np
class EnsembleND_LR(BaseEstimator):
    def __init__(self, number_of_nds, number_of_classes, penalty='l2', C=1.0, generator_String='random', random_state=1):
        self.number_of_nds = number_of_nds
        self.number_of_classes=number_of_classes
        self.model_type=LogisticRegression
        self.kwargs={'penalty': penalty, 'C': C}
        self.generator_str = generator_String
        self.params = {'number_of_nds': number_of_nds, 'number_of_classes': number_of_classes, 'penalty': penalty, 'C': C, 'generator_String': generator_String, 'random_state': random_state}
        self.random_state=random_state
                    
    def fit(self, X, y):
        # for CalibratedClassifierCV
        self.classes_=np.unique(y)
        
        # generate dichotomies
        self.forest = []
        random.seed(self.random_state)
        tree_rnd_states = random.sample(range(10000000), k=self.number_of_nds)
        lbs = list(range(self.number_of_classes))
        if (self.generator_str=='random'):
            for i in range(self.number_of_nds):
                nd2d = nd.RandomGeneration.generate(self.number_of_classes, labels=lbs, seed=tree_rnd_states[i])
                root = nd.NestedDichotomy.parse(nd2d)
                self.forest.append(root)
        elif (self.generator_str=='random_pair'):
            for i in range(self.number_of_nds):
                #print('train', i)
                nd2d = nd.RPND.generate(X, y, self.model_type, seed=tree_rnd_states[i], **self.kwargs)
                root = nd.NestedDichotomy.parse(nd2d)
                self.forest.append(root)

        # actually train model
        for i in range(self.number_of_nds):
            nd.NestedDichotomy.train(self.forest[i], X, y, self.model_type,**self.kwargs)
        
            
    def predict_proba(self, X):
        prob_preds = []
        for i in range(self.number_of_nds):
            #print('predict', i)
            y_pred = nd.NestedDichotomy.predict_proba(self.forest[i], X, self.number_of_classes)
            prob_preds.append(y_pred)
        return sum(prob_preds)/self.number_of_nds
    
    def get_params(self, deep=False):
        return self.params
    
    def set_params(self, **params):
        #print(params)
        self.kwargs = params;
        self.params['penalty'] = params['penalty']
        self.params['C'] = params['C']
        return self