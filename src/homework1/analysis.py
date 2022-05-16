import sklearn
from sklearn.metrics import classification_report,f1_score,roc_curve,auc
from numpy import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib

def kFoldValid(model,data,kfoldnums, features:list,labels:list,criterion = f1_score,shuffle=True,random_state = 12,reportEachFold=False):
    kf = sklearn.model_selection.KFold(kfoldnums,shuffle,random_state)
    X = data[features]
    Y = data[labels]
    labelTypes = [] # FIXME: label types assign
    metrics = []
    for train,valid in kf.split(data):
        train_x=X.iloc[train,:]
        train_y=Y.iloc[train]
        model.fit(train_x,train_y)
        valid_x = X.iloc[valid,:]
        valid_y = Y.iloc[valid]
        predict_y = model.predict(valid_x)
        metrics.append(criterion(valid_y,predict_y))
        if reportEachFold:
            classification_report(valid_y,predict_y)
            cm = confusion_matrix(y_true=valid_y,y_pred=predict_y)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=labelTypes)
            disp.plot()
            plt.show()
            score_y = model.decision_function(valid_x) # TODO: finish roc curves

    return mean(metrics)