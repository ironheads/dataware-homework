from copy import deepcopy
from typing import Optional
import sklearn
import matplotlib
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def encodeFeatures(data: pd.DataFrame, features: Optional[list] = None):
    data = deepcopy(data)
    if features is None:
        features = list(data.columns.values[:-1])
    dictionary = {}
    for feature in features:
        if (str(data[feature].dtype) == 'object'):
            feature_dict = data[feature].unique().tolist()
            data[feature] = data[feature].apply(
                lambda x: feature_dict.index(x))
            dictionary[feature] = feature_dict
    return data, dictionary


def encodeLabel(data: pd.DataFrame, label=None):
    data = deepcopy(data)
    if label == None:
        label = data.columns.values[-1]
    encoder = preprocessing.LabelEncoder()
    data[label] = encoder.fit_transform(data[label])
    return data, encoder


def fixEmpty(data, features: Optional[list] = None, strategy='median'):
    if features is None:
        features = list(data.columns.values[:-1])
    strategy_map = {
        'median': fillWithMedian,
        'mean': fillWithMean,
        'drop': dropEmpty
    }
    if strategy not in strategy_map:
        raise ValueError("strategy don't exist")
    data = strategy_map[strategy](data, features)
    return data


def dropEmpty(data: pd.DataFrame, features: Optional[list] = None):
    if features is None:
        features = list(data.columns.values[:-1])
    data = deepcopy(data)
    data = data.dropna(axis=0, how='any', subset=features)
    return data


def fillWithMedian(data, features: list):
    MedianImputer = SimpleImputer(missing_values=np.nan, strategy="median")
    data = deepcopy(data)
    data[features] = MedianImputer.fit_transform(data[features])
    return data


def fillWithMean(data, features: list):
    MedianImputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    data = deepcopy(data)
    data[features] = MedianImputer.fit_transform(data[features])
    return data


def standardScale(data: pd.DataFrame, features: list):
    data = deepcopy(data)
    scaler = preprocessing.StandardScaler()
    data[features] = scaler.fit_transform(data[features])
    return data


def minMaxScale(data: pd.DataFrame, features: list):
    data = deepcopy(data)
    scaler = preprocessing.MinMaxScaler()
    data[features] = scaler.fit_transform(data[features])
    return data

def scaleFeatures(data:pd.DataFrame, features:Optional[list],strategy:str = 'std'):
    if features is None:
        features = list(data.columns.values[:-1])
    strategy_map = {
        'minmax': minMaxScale,
        'std': standardScale,
        'none': lambda x: x
    }
    if strategy not in strategy_map:
        raise ValueError("strategy don't exist")
    data = strategy_map[strategy](data, features)
    return data
    
def preprocess(data:pd.DataFrame,features:Optional[list]=None,label=None,imputeStrategy:str='median',scaleStrategy:str='std'):
    data,features_encoder = encodeFeatures(data,features)
    data, label_encoder=encodeLabel(data,label)
    data = fixEmpty(data,features,imputeStrategy)
    data = scaleFeatures(data,features,scaleStrategy)
    return data,features_encoder,label_encoder

if __name__ == '__main__':
    import os
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(
        projectPath, 'dataset', "classification.csv"))
    # data,fe,le = preprocess(data)
    # print(data)
    data = dropEmpty(data)
    print(data)
    # print(fillWithMedian(data1,list(data1.columns.values[:-1])))
    # print(data)
    # convertLabelToNumber(data)
