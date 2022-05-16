import sklearn
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt

if __name__ == '__main__':
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(projectPath,'dataset',"classification.csv"))
    labels = list(data.columns.values)
    # print(list(data['治疗方案'].unique()))
    