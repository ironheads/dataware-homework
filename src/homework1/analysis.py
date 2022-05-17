import sklearn
from sklearn import metrics
from sklearn.metrics import classification_report,f1_score,roc_curve,auc
from numpy import mean
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import matplotlib
from itertools import cycle
import numpy as np
from sklearn.preprocessing import LabelEncoder,label_binarize
from sklearn.model_selection import KFold

def kFoldValid(model, X, Y ,kfoldnums ,label_encoder: LabelEncoder ,shuffle=True,random_state = 12,reportEachFold=True):
    kf = KFold(kfoldnums,shuffle=shuffle,random_state=random_state)
    labelTypes = label_encoder.classes_
    Y = label_binarize(Y,classes=range(len(labelTypes)))
    n_classes = Y.shape[1]
    metrics = []
    for train,valid in kf.split(data):
        train_x=X[train,:]
        train_y=Y[train,:]
        model.fit(train_x,train_y)
        valid_x = X[valid,:]
        valid_y = Y[valid,:]
        predict_y = model.predict(valid_x)
        id_valid_y = np.argmax(valid_y,axis=1)
        id_predict_y = np.argmax(predict_y,axis=1)
        metrics.append(f1_score(id_valid_y,id_predict_y,average='weighted'))
        if reportEachFold:
            print(classification_report(id_valid_y,id_predict_y))
            # print(id_predict_y)
            cm = confusion_matrix(y_true=id_valid_y,y_pred=id_predict_y)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=range(n_classes))
            disp.plot()
            plt.show()
            score_y = model.decision_function(valid_x)
            # Compute ROC curve and ROC area for each class
            fpr = dict()
            tpr = dict()
            roc_auc = dict()
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(valid_y[:, i], score_y[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
            # Compute micro-average ROC curve and ROC area
            fpr["micro"], tpr["micro"], _ = roc_curve(valid_y.ravel(), score_y.ravel())
            roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
            # First aggregate all false positive rates
            all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

            # Then interpolate all ROC curves at this points
            mean_tpr = np.zeros_like(all_fpr)
            for i in range(n_classes):
                mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])

            # Finally average it and compute AUC
            mean_tpr /= n_classes

            fpr["macro"] = all_fpr
            tpr["macro"] = mean_tpr
            roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

            # Plot all ROC curves
            plt.figure()
            lw=2
            plt.plot(
                fpr["micro"],
                tpr["micro"],
                label="micro-average ROC curve (area = {0:0.2f})".format(roc_auc["micro"]),
                color="deeppink",
                linestyle=":",
                linewidth=4,
            )

            plt.plot(
                fpr["macro"],
                tpr["macro"],
                label="macro-average ROC curve (area = {0:0.2f})".format(roc_auc["macro"]),
                color="navy",
                linestyle=":",
                linewidth=4,
            )

            colors = cycle(["aqua", "darkorange", "cornflowerblue"])
            for i, color in zip(range(n_classes), colors):
                plt.plot(
                    fpr[i],
                    tpr[i],
                    color=color,
                    lw=lw,
                    label="ROC curve of class {0} (area = {1:0.2f})".format(i, roc_auc[i]),
                )

            plt.plot([0, 1], [0, 1], "k--", lw=lw)
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel("False Positive Rate")
            plt.ylabel("True Positive Rate")
            plt.title("ROC curve of this Classifier")
            plt.legend(loc="lower right")
            plt.show()

    return mean(metrics)


if __name__ == '__main__':
    import os
    from preprocessing import preprocess 
    from feature_selection import selectFeatures
    from sklearn import svm
    import pandas as pd
    from sklearn.multiclass import OneVsRestClassifier
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(
        projectPath, 'dataset', "classification.csv"))
    # print(data.shape)
    data,_,lb =preprocess(data,scaleStrategy='std')
    # print(data.shape)
    X,Y = selectFeatures(data,K=10)
    classifier = OneVsRestClassifier(
        svm.SVC(kernel="linear", probability=True, random_state=0)
    )
    # kFoldValid(classifier,data,5,list(data.columns.values[:-1]),data.columns.values[-1],lb)
    kFoldValid(classifier,X,Y,3,lb)
