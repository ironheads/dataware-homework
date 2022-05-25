import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class knn(BaseEstimator, ClassifierMixin):
    def __init__(self, n_neighbors=5):
        self.n_neighbors = n_neighbors
        self.fit_flag = False

    def fit(self, x, y):
        self.train_x = x
        self.train_y = y
        self.fit_flag = True

    def predict(self, test_x):
        if not self.fit_flag:
            print("must use fit to set ground truth")
            return

        num_test = test_x.shape[0]
        dis_array = self.compute_dis(test_x)

        y_pred = np.zeros(num_test)
        for i in range(num_test):
            dist_k_min = np.argsort(dis_array[i])[:self.n_neighbors]
            k_label = self.train_y[dist_k_min]
            y_pred[i] = np.argmax(np.bincount(k_label.tolist()))
        return y_pred

    def predict_proba(self, test_x):
        if not self.fit_flag:
            print("must use fit to set ground truth")
            return
        num_test = test_x.shape[0]
        dis_array = self.compute_dis(test_x)

        y_pred = []
        for i in range(num_test):
            label_per_class = np.zeros(3)
            dist_k_min = np.argsort(dis_array[i])[:self.n_neighbors]
            k_label = self.train_y[dist_k_min]
            for label in k_label:
                label_per_class[label] += 1
            y_pred.append(label_per_class)
        y_pred = np.array(y_pred)
        y_pred /= self.n_neighbors
        return y_pred

    def compute_dis(self, test_x):
        if not self.fit_flag:
            print("must use fit to set ground truth")
            return

        #(test_x - train_x)^2 = test_x^2 + train_x^2 - 2 * test_x * train_x
        test2 = np.sum(np.square(test_x), axis=1, keepdims=True) # test * 1
        train2 = np.sum(np.square(self.train_x), axis=1) # train
        mix2 = -2 * np.dot(test_x, self.train_x.T) # test * train
        dis = np.sqrt(mix2 + test2 + train2)
        return dis