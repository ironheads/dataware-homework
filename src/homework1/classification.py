import sklearn
import pandas as pd
import matplotlib
import os
import matplotlib.pyplot as plt
from preprocessing import preprocess,dropEmpty
from feature_selection import selectFeatures
from analysis import kFoldValid
from sklearn.svm import LinearSVC
import argparse
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import  AdaBoostClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='classification')
    parser.add_argument('--without_preprocessing', action='store_true', help='without preprocessing')
    parser.add_argument('--without_feature_selection', action='store_true', help='without feature selection')
    parser.add_argument('--model', type=str, choices=['SVC', 'KNC','ABC'], default='SVC', help="choose which model to use")
    parser.add_argument('--num_features',type=int, default=12,help="the number of features reserved to fit the model")
    parser.add_argument('--num_fold',type=int,default=5,help="the K Fold Number K")
    args = parser.parse_args()
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(projectPath,'dataset',"classification.csv"))
    features = list(data.columns.values[:-1])
    label = data.columns.values[-1]
    if not args.without_preprocessing:
        data,feature_dict,label_encoder = preprocess(data,features,label)
    else:
        data = dropEmpty(data,features)
    if not args.without_feature_selection:
        X,Y=selectFeatures(data,features,label,K=args.num_features)
    else:
        X,Y = data[features].values,data[label].values
    
    model_map = {
        'SVC': LinearSVC(random_state=1234),
        'KNC': KNeighborsClassifier(),
        'ABC': AdaBoostClassifier(random_state=1234)
    }
    # print(X)
    # print(Y)
    
    model = OneVsRestClassifier(model_map[args.model])
    print(kFoldValid(model,X,Y,args.num_fold,label_encoder))
    # print(list(data['治疗方案'].unique()))
    