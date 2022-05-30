from copy import deepcopy
import numpy as np
import sklearn
import pandas as pd
from typing import Optional
from sklearn.feature_selection import SelectKBest,chi2,mutual_info_classif
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import RFE,SelectFromModel
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import LinearSVC



def selectFeatures(data:pd.DataFrame,features:Optional[list]=None,label=None,strategy='filter',method='mic',K=5,embedded=False):
    if features is None:
        features=list(data.columns.values[:-1])
    if label is None:
        label=data.columns.values[-1]
    X=data[features]
    Y=data[label]
    # X = dropFeatures(X)
    if embedded:
        X=embeddedFeatures(X,Y)
    strategy_map = {
        'filter': filterFeatures,
        'wrap': wrapFeatures,
    }
    return strategy_map[strategy](X=X,Y=Y,strategy=method,K=K), Y.values

def embeddedFeatures(X,Y,strategy:str="SVC"):
    strategy_map = {
        'SVC': SelectFromModel(LinearSVC(C=0.01,penalty="l1", dual=False).fit(X,Y),prefit=True)
    }
    return strategy_map[strategy].transform(X)

def filterFeatures(X,Y,strategy:str='chi2',K=5):
    strategy_map = {
        'mic': mutual_info_classif,
        'chi2': chi2,
    }
    return SelectKBest(strategy_map[strategy],k=K).fit_transform(X,Y)



def wrapFeatures(X,Y,strategy='RFE',K=5):
    strategy_map = {
        'RFE': RFE(estimator=RidgeClassifier(),n_features_to_select=K),
    }
    return strategy_map[strategy].fit_transform(X,Y)

def dropFeatures(X):
    X = deepcopy(X)
    selector = VarianceThreshold()
    X = selector.fit_transform(X)
    # X = VarianceThreshold(np.median(X.var().values)).fit_transform(X)
    return X

if __name__ == '__main__':
    import os
    from preprocessing import preprocess
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(
        projectPath, 'dataset', "classification.csv"))
    # print(data.shape)
    data,_,_ =preprocess(data,scaleStrategy='std')
    X,Y = selectFeatures(data,K=10)
    # print(data.shape)
