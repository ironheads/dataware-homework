from cmath import inf
import numpy as np
from typing import Callable, Optional,Union
from sklearn.utils import check_random_state
from numpy import random
import math
import warnings

class KmeansClustering(object):
    def __init__(self,
                n_clusters:int = 8,
                *,
                init:Union[str,np.ndarray,Callable] = 'k-means++',
                max_iter:int = 300,
                verbose: int = 0,
                random_state: Union[np.random.RandomState,int] = None,
                algorithm: str = 'full',
                affinity: str = 'eculidean_square'
                ) -> None:
        if algorithm == 'auto':
            algorithm == 'full'
        self.affinity = affinity
        self.n_clusters = n_clusters
        self.init = init
        self.max_iter = max_iter
        self.verbose = verbose
        self.random_state = check_random_state(random_state)
        self.algorithm = algorithm
        if self.algorithm != 'full':
            raise ValueError('only support full algorithm')

    @property
    def _distance_function(self):
        if callable(self.affinity):
            return self.affinity
        else:
            affinity_map = {
                'eculidean_square': lambda x,y : np.sum(np.power(x-y,2)) ,
                'l1': lambda x,y: np.linalg.norm(x-y,ord=1),
                'l2': lambda x,y: np.linalg.norm(x-y,ord=2),
                'manhattan': lambda x,y: np.linalg.norm(x-y,ord=1),
                'cosine': lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y))),
            }
            if self.affinity in affinity_map:
                return affinity_map[self.affinity]
            else:
                raise ValueError('affinity value error')

    #对一个样本找到与该样本距离最近的聚类中心
    def _nearest(self, point, cluster_centers):
        min_dist = inf
        m = np.shape(cluster_centers)[0]  # 当前已经初始化的聚类中心的个数
        for i in range(m):
            # 计算point与每个聚类中心之间的距离
            d = self._distance_function(point, cluster_centers[i, ])
            # 选择最短距离
            if min_dist > d:
                min_dist = d
        return min_dist

    def _initialized_centers(self,X,sample_weight):
        if callable(self.init):
            return self.init(X,self.n_clusters,self.random_state)
        elif self.init=='k-means++':
            n_samples,n_features = X.shape
            cluster_centers = np.zeros((self.n_clusters,n_features))
            new_center_index = self.random_state.randint(0,n_samples)
            cluster_centers[0,]=X[new_center_index,]
            # 2、初始化一个距离的序列
            d = [0.0 for _ in range(n_samples)]
            for i in range(1, self.n_clusters):
                sum_all = 0
                for j in range(n_samples):
                    # 3、对每一个样本找到最近的聚类中心点
                    d[j] = self._nearest(X[j, ], cluster_centers[0:i, ])
                    # 4、将所有的最短距离相加
                    sum_all += d[j]
                # 5、取得sum_all之间的随机值
                sum_all *= self.random_state.rand()
                # 6、获得距离最远的样本点作为聚类中心点
                for j, di in enumerate(d):
                    sum_all=sum_all - di
                    if sum_all > 0:
                        continue
                    cluster_centers[i,] = X[j, ]
                    break
            return cluster_centers

        elif self.init=='random':
            n_samples,_ = X.shape
            sample_list = self.random_state.choice(n_samples,self.n_clusters,replace=False,p=sample_weight)
            return np.copy(X[sample_list,:])

        elif isinstance(self.init,np.ndarray):
            _,n_features = X.shape
            x,y = self.init.shape
            if x!=self.n_clusters or y!=n_features:
                raise ValueError("init ndarray should be shape (%d,%d) but is shape(%d,%d)"%(self.n_clusters,n_features,x,y))
            return self.init

    @staticmethod
    def _findClostestCentroids(X, centroid):
        idx = np.zeros((np.size(X, 0)), dtype=int)
        n = X.shape[0]  # n 表示样本个数
        inertia = 0.0
        for i in range(n):
            subs = centroid - X[i, :]
            dimension2 = np.power(subs, 2)
            dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            dimension_s = np.nan_to_num(dimension_s)
            min_value = dimension_s.min()
            inertia += min_value
            idx[i] = np.where(dimension_s == min_value)[0][0]
        return idx, inertia

    @staticmethod
    def _cluster_distance(X,centroid):
        n_samples,n_clusters = np.size(X, 0),np.size(centroid,0)
        res = np.zeros((n_samples,n_clusters), dtype=int)
        n = X.shape[0]  # n 表示样本个数
        for i in range(n):
            subs = centroid - X[i, :]
            dimension2 = np.power(subs, 2)
            dimension_s = np.sum(dimension2, axis=1)  # sum of each row
            dimension_s = np.nan_to_num(dimension_s)
            res[i,]=dimension_s
        return res

    @staticmethod
    def _computeCentroids(X, idx, K):
        n, m = X.shape
        centriod = np.zeros((K, m), dtype=float)
        for k in range(K):
            index = np.where(idx == k)[0]  # 一个簇一个簇的分开来计算
            temp = X[index, :]  # ? by m # 每次先取出一个簇中的所有样本
            s = np.sum(temp, axis=0)
            centriod[k, :] = s / np.size(index)
        return centriod
    
    def fit(self,X,y=None,sample_weight=None):
        n_samples,n_features = X.shape
        if n_samples < self.n_clusters:
            raise ValueError("n_samples=%d should be >= n_clusters=%d" % (
                n_samples, self.n_clusters))
        if sample_weight is not None:
            raise ValueError('do not support sample weight')
        cluster_centers = self._initialized_centers(X,sample_weight)
        idx, inertia = self._findClostestCentroids(X,cluster_centers)
        best_labels,best_inertia,best_cluster_centers,best_iter=idx,inertia,cluster_centers,0
        for i in range(1,self.max_iter+1):
            idx, inertia = self._findClostestCentroids(X,cluster_centers)
            cluster_centers = self._computeCentroids(X, idx, self.n_clusters)
            if inertia < best_inertia:
                best_labels=idx.copy()
                best_inertia=inertia
                best_cluster_centers=cluster_centers.copy()
                best_iter=i
        self.cluster_centers_ = best_cluster_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_iter
        return self

    def fit_predict(self,X,y=None,sample_weight=None):
        return self.fit(X,y,sample_weight).labels_

    def predict(self,X,sample_weight=None):
        if sample_weight is not None:
            raise ValueError('do not support sample weight')
        labels, _ = self._findClostestCentroids(X,self.cluster_centers_)
        return labels

    def transform(self,X):
        return self._cluster_distance(X,self.cluster_centers_)

    def fit_transform(self,X,y=None,sample_weight=None):
        if sample_weight is not None:
            raise ValueError('do not support sample weight')
        self.fit(X,y,sample_weight)
        return self.transform(X)

    def score(self, X, y=None, sample_weight=None):
        warnings.warn('score function do not support')
        idx, inertia = self._findClostestCentroids(X,self.cluster_centers_)
        return -inertia

if __name__ == '__main__':
    import numpy as np
    import os
    import pandas as pd
    clustering = KmeansClustering(10,max_iter=300)
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(projectPath,'dataset',"cluster.csv"))
    X = data[1:-1].values
    labels = clustering.fit_predict(X)
    print(labels)