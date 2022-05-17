import numpy as np
import sklearn
import pandas as pd
from typing import Optional
from scipy.stats import pearsonr
from minepy import MINE
from sklearn.feature_selection import SelectKBest,chi2
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import RFE,SelectFromModel

'''
 FIXME : lots of problems 
 need to read and modify more
'''
def selectFeatures(data:pd.DataFrame,features:Optional[list]=None,label=None,strategy='filter',method='pearsonr',K=5):
    if features is None:
        features=list(data.columns.values[:-1])
    if label is None:
        label=data.columns.values[-1]
    strategy_map = {
        'filter': filterFeatures,
        'wrap': wrapFeatures
    }
    return strategy_map[strategy](data=data,features=features,label=label,strategy=method,K=K)


def filterFeatures(data:pd.DataFrame,features:Optional[list],label,strategy:str='chi2',K=5):
    def mic(x, y):
        m = MINE()
        m.compute_score(x, y)
        return (m.mic(), 0.5)
    strategy_map = {
        'pearsonr': lambda X, Y: np.array(map(lambda x:pearsonr(x, Y), X.T)).T,
        'chi2': chi2,
        'mic': lambda X, Y: np.array(map(lambda x:mic(x, Y), X.T)).T
    }
    return SelectKBest(strategy_map[strategy],k=K).fit_transform(data[features],data[label])



def wrapFeatures(data:pd.DataFrame,features:Optional[list],label,strategy='RFE',K=5):
    strategy_map = {
        'RFE': RFE(estimator=SGDClassifier(),n_features_to_select=K),
    }
    return strategy_map[strategy].fit_transform(data[features],data[label])


if __name__ == '__main__':
    import os
    from preprocessing import preprocess
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(
        projectPath, 'dataset', "classification.csv"))
    data,_,_ =preprocess(data,scaleStrategy='minmax')
    data = selectFeatures(data)
    print(data)