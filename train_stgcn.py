# -*- coding: utf-8 -*-
"""
Created on Tue Jul 21 16:43:11 2020

@author: Ming Jin
"""

import dgl
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.preprocessing import StandardScaler

from utils import get_adjacency_matrix, data_transform
from network import STGCN

"""
Hyperparameters

"""
# training params
lr = 0.001
batch_size = 64
epochs = 50
window = 12
horizon = 3
drop_prob = 0.3
save_path = './checkpoints/stgcn.pt'

# Device params
DisableGPU = False
device = torch.device("cuda") if torch.cuda.is_available() and not DisableGPU else torch.device("cpu")

# model params
model_structure = 'TSTNTSTN'  # OutputLayer will be added automatically after this
channels = [1, 64, 16, 64, 64, 16, 64]  # design for both temporal and spatial conv

# dataset params
sensor_ids = './data/METR-LA/graph_sensor_ids.txt'
sensor_distance = './data/METR-LA/distances_la_2012.csv'
recording = './data/METR-LA/metr-la.h5'


"""
Data preprocessing

"""
# read sensor IDs
with open(sensor_ids) as f:
    sensor_ids = f.read().strip().split(',')

# read sensor distance
distance_df = pd.read_csv(sensor_distance, dtype={'from': 'str', 'to': 'str'})

# build adj matrix based on equation (10)
adj_mx = get_adjacency_matrix(distance_df, sensor_ids)

# transform adj_mx to scipy.sparse.coo_matrix
# a sparse matrix in coordinate format
sp_mx = sp.coo_matrix(adj_mx)

# construct DGLGraph based on sp_mx (adj_mx)
G = dgl.DGLGraph()
G.from_scipy_sparse_matrix(sp_mx)

# read & process time series recording
df = pd.read_hdf(recording)
num_samples, num_nodes = df.shape

len_train = round(num_samples * 0.7)
len_val = round(num_samples * 0.1)
train = df[: len_train]
val = df[len_train: len_train + len_val]
test = df[len_train + len_val:]

# del zero rows from train, val, and test
train = train[~(train == 0).all(axis=1)]
val = val[~(val == 0).all(axis=1)]
test = test[~(test == 0).all(axis=1)]

scaler = StandardScaler()
train = scaler.fit_transform(train)
val = scaler.transform(val)
test = scaler.transform(test)

# x with the shape [:, 1, window, num_nodes] where 1 means the channel
# y with the shape [:, num_nodes]
x_train, y_train = data_transform(train, window, horizon, device)
x_val, y_val = data_transform(val, window, horizon, device)
x_test, y_test = data_transform(test, window, horizon, device)

train_data = TensorDataset(x_train, y_train)
train_iter = DataLoader(train_data, batch_size, shuffle=True)
val_data = TensorDataset(x_val, y_val)
val_iter = DataLoader(val_data, batch_size)
test_data = TensorDataset(x_test, y_test)
test_iter = DataLoader(test_data, batch_size)


"""
STGCN Training

"""
# create a network instance
model = STGCN(channels, window, num_nodes, G, drop_prob, model_structure).to(device)

# define loss and optimizer
loss = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

def evaluate_model(model, loss, data_iter):
    model.eval()
    l_sum, n = 0.0, 0
    with torch.no_grad():
        for x, y in data_iter:
            y_pred = model(x).view(len(x), -1)
            l = loss(y_pred, y)
            l_sum += l.item() * y.shape[0]
            n += y.shape[0]
        return l_sum / n

for epoch in range(1, epochs + 1):
    l_sum, n = 0.0, 0  # 'l_sum' is epoch sum loss, 'n' is epoch instance number
    model.train()
    for x, y in tqdm.tqdm(train_iter):
        y_pred = model(x).view(len(x), -1)  # [batch_size, num_nodes]
        l = loss(y_pred, y)
        optimizer.zero_grad()
        l.backward()
        optimizer.step()
        l_sum += l.item() * y.shape[0]
        n += y.shape[0]
    scheduler.step()
    val_loss = evaluate_model(model, loss, val_iter)  # epoch validation
    # GPU mem usage
    gpu_mem_alloc = torch.cuda.max_memory_allocated() / 1000000 if torch.cuda.is_available() else 0
    # save model every epoch
    torch.save(model.state_dict(), save_path)
    # print("epoch", epoch, ", train loss:", l_sum / n, ", validation loss:", val_loss)
    print('Epoch {:03d} | lr {:.6f} |Train Loss {:.5f} | Val Loss {:.5f} | GPU {:.1f} MiB'.format(
        epoch, optimizer.param_groups[0]['lr'], l_sum / n, val_loss, gpu_mem_alloc))

print('\nTraining finished.\n')

"""
STGCN Testing

"""
# calculate MAE, MAPE, and RMSE 
def evaluate_metric(model, data_iter, scaler):
    model.eval()
    with torch.no_grad():
        mae, mape, mse = [], [], []
        for x, y in data_iter:
            y = scaler.inverse_transform(y.cpu().numpy()).reshape(-1)
            y_pred = scaler.inverse_transform(model(x).view(len(x), -1).cpu().numpy()).reshape(-1)
            d = np.abs(y - y_pred)
            mae += d.tolist()
            mape += (d / y).tolist()
            mse += (d ** 2).tolist()
        MAE = np.array(mae).mean()
        MAPE = np.array(mape).mean()
        RMSE = np.sqrt(np.array(mse).mean())
        return MAE, MAPE, RMSE
    
test_MSE = evaluate_model(model, loss, test_iter)
print('Test loss {:.6f}'.format(test_MSE))
test_MAE, test_MAPE, test_RMSE = evaluate_metric(model, test_iter, scaler)
print('MAE {:.5f} | MAPE {:.5f} | RMSE {:.5f}'.format(
    test_MAE, test_MAPE, test_RMSE))