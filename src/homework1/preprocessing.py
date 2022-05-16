from copy import deepcopy
from traceback import print_tb
from typing import Optional
import sklearn
import matplotlib
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer


def fillEmpty(data,features:Optional[list]=None):
    data = deepcopy(data)
    if features is None:
        features = data.columns.values[:-1]

    return data

def dropEmpty(data,features:Optional[list]=None):
    pass 
def fillWithMedian(data,features:list):
    MedianImputer =SimpleImputer(missing_values=np.nan, strategy="median")
    data = deepcopy(data)
    data[features] = MedianImputer.fit_transform(data[features])
    return data


if __name__ == '__main__':
    import os
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(projectPath,'dataset',"classification.csv"))
    print(fillWithMedian(data,list(data.columns.values[:-1])))