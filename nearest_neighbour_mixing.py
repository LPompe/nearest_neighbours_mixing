#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 14:33:17 2020

@author: Lucas Pompe, Institute of neuroinformatics, UZH/ETH 


"""
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import MDS



def mixing_matrix(k, X, y=None, metric='euclidean', algorithm='brute', 
                  n_jobs=-1, **kwargs):
    """
    create mixing matrix 
    
    To avoid a massive swarm of arguments in this function you should
    figure out how to plot the matrix yourself.
    
    

    Parameters
    ----------
    k : int,
        number of neighbours to find
    
    X : array-like,
        'training' matrix
    
    y : array-like or None,
        matrices for which neighbours will be found. If None, X will be used 
        instead
        
    metric : String or mapable,
            distance function for KNN, refer to SKLEARN docs
        
    algorithm : String,
            KNN algorithm, refer to SKLEARN docs
        
    n_jobs : int,
        How many workers to use, refer to SKLEARN docs
        
    **kwargs : 
        additional arguments to pass to SKLEARN NN search
    """
    
    nn = create_nn(k, X, metric, algorithm, n_jobs, **kwargs)
    
    if y is None:
        y = X
        
    neighbour_idcs = nn.kneighbors(y, 
                                   return_distance=return_distance)
    
    
    
    idcs = np.arange(len(y))
    raveled_neighbour_idcs = np.ravel(neighbour_idcs)
    
    
    
    hist = np.histogram2d(idcs, raveled_neighbour_idcs, bins=(len(y), k))
    return hist
    
    
    
    
def create_nn(k, X, metric='euclidean', algorithm='brute', 
              n_jobs=-1, **kwargs):
    """
    fit a nearest neighbours routine to X, 
    
    kwargs will be passed to knn
    """
    nn = NearestNeighbors(k, metric=metric, algorithm=algorithm, 
                        n_jobs=n_jobs, **kwargs)
    
    nn = nn.fit(X)
    return nn
        
    
def mds():
    """
    Non-standard multidimensional scaling with custom dissimilarity matrix
    """
    raise NotImplementedError()
        
        

        



