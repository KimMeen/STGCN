# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 17:23:10 2020

@author: Ming Jin
"""

import torch
import numpy as np


def get_adjacency_matrix(distance_df, sensor_ids, normalized_k=0.1):
    
    """
    Equation (10) in the paper
    
    
    Parameters
    ----------
    distance_df : DataFrame
        Sensor distances, data frame with three columns: [from, to, distance]
    sensor_ids : List
        List of sensor ids.
    normalized_k : Int
        Entries that become lower than normalized_k after normalization are set to zero for sparsity

    Returns
    -------
    Array
        Adjacency matrix.
    """
    
    # init dist_mx
    num_sensors = len(sensor_ids)
    dist_mx = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    dist_mx[:] = np.inf
    
    # builds sensor id to index map
    sensor_id_to_ind = {}
    for i, sensor_id in enumerate(sensor_ids):
        sensor_id_to_ind[sensor_id] = i
    
    # fills cells in the dist_mx with distances
    for row in distance_df.values:
        if row[0] not in sensor_id_to_ind or row[1] not in sensor_id_to_ind:
            continue
        dist_mx[sensor_id_to_ind[row[0]], sensor_id_to_ind[row[1]]] = row[2]

    # calculates the standard deviation as theta.
    distances = dist_mx[~np.isinf(dist_mx)].flatten()  # get all valid distances
    std = distances.std()
    # variation of equation (10) in the paper
    adj_mx = np.exp(-np.square(dist_mx / std))
    # make the adjacent matrix symmetric by taking the max
    # adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])

    # sets entries that lower than a threshold, i.e., k, to zero for sparsity
    # normalized_k is epsilon in equation (10) to control the sparsity 
    adj_mx[adj_mx < normalized_k] = 0
    
    return adj_mx


def data_transform(data, n_his, n_pred, device):
    """
    Produce data slices for training and testing
    The task to perform is single-step ahead forecasting
    
    Parameters
    ----------
    data : DataFrame
        Input data (e.g. Train, val, and test)
    n_his : Int
        Window size of the historical observation
    n_pred : Int
        Horizon to be predicted
    device : Torch.device
        Place the data to which device

    Returns
    -------
    Torch.Tensor
        x with the shape [len(data)-n_his-n_pred, 1, n_his, num_nodes]
        y with the shape [len(data)-n_his-n_pred, num_nodes]

    """
    
    # number of nodes
    n_route = data.shape[1]
    # number of recordings
    l = len(data)
    # number of instances
    num = l-n_his-n_pred
    
    x = np.zeros([num, 1, n_his, n_route])
    y = np.zeros([num, n_route])
    
    idx = 0
    for i in range(l-n_his-n_pred):
        head = i
        tail = i + n_his
        # x is the historical observations
        x[idx, :, :, :] = data[head: tail].reshape(1, n_his, n_route)
        # y is 'n_pred', i.e. Horizon, ahead value
        y[idx] = data[tail + n_pred - 1]
        # idx from 0 to num-1
        idx += 1
        
    return torch.Tensor(x).to(device), torch.Tensor(y).to(device)