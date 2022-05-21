import numpy as np
from typing import Callable, Optional,Union
from sklearn.cluster import affinity_propagation

from torch import affine_grid_generator, narrow, per_channel_affine

class AgglomerativeClustering(object):
    def __init__(self, 
                n_clusters:Optional[int] = 2,
                *,
                affinity:Union[str,Callable] = 'euclidean',
                memory = None,
                connectivity=None, 
                compute_full_tree='auto', 
                linkage:str='ward', 
                distance_threshold=None
                ) -> None:
        self.n_clusters = n_clusters
        self.affinity=affinity
        self.memory=memory
        self.connectivity=connectivity
        self.compute_full_tree = compute_full_tree
        self.linkage=linkage
        self.distance_threshold=distance_threshold
    

    def fit(self,X,y=None):
        if self.n_clusters is not None and self.distance_threshold is not None:
            raise RuntimeError('assign n_clusters and distance_threshold at the same time.')
        if self.n_clusters is None and self.distance_threshold is None:
            raise RuntimeError('do not know when the clustering ends')
        if self.linkage == 'ward' and self.affinity != 'euclidean':
            raise RuntimeError('only euclidean distance can be used when the linkage is ward')

        if self.n_clusters is not None:
            return self._fit_by_n_clusters(X,y)
        else:
            return self._fit_by_distance_threshold(X,y)

    def _fit_by_distance_threshold(self,X,y=None):
        n_samples = len(X)
        distance_threshold = self.distance_threshold
        ## initialize 
        sets = [set(i) for i in range(n_samples)]
        pass

    def _fit_by_n_clusters(self,X,y=None):
        n_samples = len(X)
        n_clusters = self.n_clusters
        ## initialize 
        sets = [set(i) for i in n_clusters:

        pass

    def _precompute_distance(self,X):
        n_samples = len(X)
        distance_matrix = np.zeros((n_samples,n_samples))
        
        pass
    

    def _distance_function(self):
        if callable(self.affinity):
            return self.affinity
        else:
            affinity_map = {
                'euclidean': lambda x,y : np.linalg.norm(x-y),
                'l1': lambda x,y: np.linalg.norm(x-y,ord=1),
                'l2': lambda x,y: np.linalg.norm(x-y,ord=2),
                'manhattan': lambda x,y: np.linalg.norm(x-y,ord=1),
                'cosine': lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y)))
                'percomputed'
            }