from cmath import inf
import numpy as np
from typing import Callable, Optional,Union

class AgglomerativeClustering(object):
    def __init__(self, 
                n_clusters:Optional[int] = 2,
                *,
                affinity:Union[str,Callable] = 'eculidean',
                memory = None,
                connectivity=None, 
                compute_full_tree=None, 
                linkage:str='ward', 
                distance_threshold=None
                ) -> None:
        self.n_clusters = n_clusters
        self.affinity=affinity
        self.memory=memory
        if memory is not None:
            raise RuntimeError('unsupported argument memory')
        self.connectivity=connectivity
        self.compute_full_tree = compute_full_tree
        if compute_full_tree is not None:
            raise RuntimeError('unsupported argument compute_full_tree')
        self.linkage=linkage
        self.distance_threshold=distance_threshold
    

    def fit(self,X,y=None):
        if self.n_clusters is not None and self.distance_threshold is not None:
            raise RuntimeError('assign n_clusters and distance_threshold at the same time.')
        if self.n_clusters is None and self.distance_threshold is None:
            raise RuntimeError('do not know when the clustering ends')
        if self.linkage == 'ward' and self.affinity != 'euclidean':
            raise RuntimeError('only euclidean distance can be used when the linkage is ward')
        n_samples = len(X)
        self.children_ = np.zeros((n_samples-1,2),dtype=np.int32)
        self.labels_ = np.zeros((n_samples),dtype=np.int32)
        if self.n_clusters is not None:
            return self._fit_by_n_clusters(X,y)
        else:
            return self._fit_by_distance_threshold(X,y)

    def _fit_by_distance_threshold(self,X,y=None):
        n_samples = len(X)
        distance_threshold = self.distance_threshold
        ## initialize 
        sets = dict([(i,set([i])) for i in range(n_samples)])
        if self.affinity!='ward':
            self.connectivity=self._precompute_distance(X)
        num_branch = n_samples
        while len(sets)>1:
            set_min1=None
            set_min2=None
            min_distance = inf
            key_values = [key for key in sets.keys()]
            for i in range(len(key_values)-1):
                key_i = key_values[i]
                for j in range(i+1,len(key_values)):
                    key_j = key_values[j]
                    if min_distance > self._distance_between_sets(sets[key_i],sets[key_j]):
                        set_min1 = key_i
                        set_min2 = key_j
                        min_distance = self._distance_between_sets(sets[key_i],sets[key_j])
            if min_distance > distance_threshold:
                break
            sets[num_branch]=sets[set_min1] | sets[set_min2]
            sets.pop(set_min1)
            sets.pop(set_min2)
            self.children_[num_branch-n_samples,0]=set_min1
            self.children_[num_branch-n_samples,1]=set_min2
            num_branch+=1
            print(num_branch-n_samples)
        class_id = 0
        self.n_clusters_ = 2*n_samples-num_branch
        for values in sets.values():
            for id in values:
                self.labels_[id]=class_id
            class_id += 1
        y=self.labels_
        return self.labels_
        

    def _fit_by_n_clusters(self,X,y=None):
        n_samples = len(X)
        n_clusters = self.n_clusters
        ## initialize 
        sets = dict([(i,set([i])) for i in range(n_samples)])
        if self.affinity!='ward':
            self.connectivity=self._precompute_distance(X)
        num_branch = n_samples
        while len(sets)>n_clusters:
            set_min1=None
            set_min2=None
            min_distance = inf
            key_values = [key for key in sets.keys()]
            for i in range(len(key_values)-1):
                key_i = key_values[i]
                for j in range(i+1,len(key_values)):
                    key_j = key_values[j]
                    if min_distance > self._distance_between_sets(sets[key_i],sets[key_j],X):
                        set_min1 = key_i
                        set_min2 = key_j
                        min_distance = self._distance_between_sets(sets[key_i],sets[key_j],X)
            sets[num_branch]=sets[set_min1] | sets[set_min2]
            sets.pop(set_min1)
            sets.pop(set_min2)
            self.children_[num_branch-n_samples,0]=set_min1
            self.children_[num_branch-n_samples,1]=set_min2
            num_branch+=1
            print(num_branch-n_samples)
        class_id = 0
        self.n_clusters_ = n_clusters
        for values in sets.values():
            for id in values:
                self.labels_[id]=class_id
            class_id += 1
        y=self.labels_
        return self.labels_


    @property
    def _distance_between_sets(self):
        def single_link(x:set,y:set, *_):
            set_distance = inf
            for a in x:
                for b in y:
                    if set_distance > self.connectivity[a,b]:
                        set_distance = self.connectivity[a,b]
            return set_distance

        
        def complete_link(x:set,y:set, *_):
            set_distance = -inf
            for a in x:
                for b in y:
                    if set_distance < self.connectivity[a,b]:
                        set_distance = self.connectivity[a,b]
            return set_distance
        
        def average_link(x:set,y:set,*_):
            sum_distance = 0.0
            for a in x:
                for b in y:
                    sum_distance += self.connectivity[a,b]
            return sum_distance/(len(x)*len(y))

        def ward_link(x:set,y:set,X ,*_):
            m_x = np.zeros((X.shape[1]))
            m_y = np.zeros((X.shape[1]))
            for a in x:
                m_x += X[a,:]
            for b in y:
                m_y += X[b,:]
            m_x /= len(x)
            m_y /= len(y)
            return len(x)*len(y)*self._distance_function(m_x,m_y)/(len(x)+len(y))
            
        distance_map = {
            'ward': ward_link,
            'complete': complete_link,
            'average': average_link,
            'single': single_link
        }
        if self.linkage not in distance_map:
            raise ValueError('linkage value error')
        return distance_map[self.linkage]

    def _precompute_distance(self,X):
        if self.linkage == 'ward':
            return None
        if self.affinity == 'precomputed':
            return self.connectivity
        n_samples = len(X)
        distance_matrix = np.zeros((n_samples,n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                distance_matrix[i,j]=self._distance_function(X[i,:],X[j,:])
        return distance_matrix
    
    @property
    def _distance_function(self):
        if callable(self.affinity):
            return self.affinity
        else:
            affinity_map = {
                'eculidean': lambda x,y : np.linalg.norm(x-y),
                'l1': lambda x,y: np.linalg.norm(x-y,ord=1),
                'l2': lambda x,y: np.linalg.norm(x-y,ord=2),
                'manhattan': lambda x,y: np.linalg.norm(x-y,ord=1),
                'cosine': lambda x,y: np.dot(x,y)/(np.linalg.norm(x)*(np.linalg.norm(y))),
                'percomputed': lambda x,y: None
            }
            if self.affinity in affinity_map:
                return affinity_map[self.affinity]
            else:
                raise ValueError('affinity value error')

if __name__ == '__main__':
    import numpy as np
    import os
    import pandas as pd
    clustering = AgglomerativeClustering(100)
    projectPath = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    data = pd.read_csv(os.path.join(projectPath,'dataset',"cluster.csv"))
    X = data[1:-1].values
    # true_labels = np.array([0, 0, 1, 1, 2, 2])
    labels = clustering.fit(X)
    print(labels)