# -*- coding: utf-8 -*-
"""
Created on Sun Jul 12 19:37:02 2020

@author: Ming Jin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph


class GCN(nn.Module):
    '''
    in_feats:
        Input feature size
    out_feats:
        Output feature size
    activation:
        Applies an activation function to the updated node features
    '''
    
    def __init__(self, in_feats, out_feats, activation=None):
        super(GCN, self).__init__()
        self._in_feats = in_feats  # "C" in the paper
        self._out_feats = out_feats  # "F" in the paper
        self._activation_func = activation  # "ReLu" and "Softmax" in the paper
        
        # W^{1} and W^{2} in the paper
        self.weight = nn.Parameter(torch.Tensor(in_feats, out_feats))
        self.bias = nn.Parameter(torch.Tensor(out_feats))  # bias, optional
        
        # initialize the weight and bias
        self.reset_parameters()
    
        
    def reset_parameters(self):
        '''
        Reinitialize learnable parameters
        ** Glorot, X. & Bengio, Y. (2010)
        ** Critical, otherwise the loss will be NaN
        '''
        
        if self.weight is not None:
            nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)
    
            
    def forward(self, g, features):
        '''
        formular: 
            h_i^{(l+1)} = \sigma(b^{(l)} + \sum_{j\in\mathcal{N}(i)}\frac{1}{c_{ij}}h_j^{(l)}W^{(l)})
        
        Inputs:
            g: 
                The fixed graph
            features: 
                H^{l}, i.e. Node features with shape [num_nodes, features_per_node]
                
        Returns:
            rst:
                H^{l+1}, i.e. Node embeddings of the l+1 layer with the 
                shape [num_nodes, hidden_per_node]
                
        Variables:
            gcn_msg: 
                Message function of GCN, i.e. What to be aggregated 
                (e.g. Sending node embeddings)
            gcn_reduce: 
                Reduce function of GCN, i.e. How to aggregate 
                (e.g. Summing neighbor embeddings)
                
        Notice: 'h' means node feature/embedding itself, 'm' means node's mailbox
        '''
                        
        # normalize features by node's out-degrees
        out_degs = g.out_degrees().to(features.device).float().clamp(min=1)  # shape [num_nodes]
        norm1 = torch.pow(out_degs, -0.5)
        shape1 = norm1.shape + (1,) * (features.dim() - 1)
        norm1 = torch.reshape(norm1, shape1)
        features = features * norm1
        
        # mult W first to reduce the feature size for aggregation
        features = torch.matmul(features, self.weight)
        
        # DGLGraph.ndata: Data view of all the nodes (a.k.a node features), 
        # g.ndata['h'] is a dictionary, 'h' is the key (identifier)
        # i.e. {'h' : tensor(...)}
        g.ndata['h'] = features  # provide each node with its feature
        # define the message and reduce functions
        gcn_msg = fn.copy_src(src='h', out='m')
        gcn_reduce = fn.sum(msg='m', out='h')
        # message passing + update
        g.update_all(gcn_msg, gcn_reduce)
        rst = g.ndata.pop('h')
        
        # normalize features by node's in-degrees
        in_degs = g.in_degrees().to(features.device).float().clamp(min=1)  # shape [num_nodes]
        norm2 = torch.pow(in_degs, -0.5)
        shape2 = norm2.shape + (1,) * (features.dim() - 1)
        norm2 = torch.reshape(norm2, shape2)
        rst = rst * norm2
        
        # add bias
        rst = rst + self.bias
        
        # activation
        if self._activation_func is not None:
            rst = self._activation_func(rst)
        
        # get h^{l+1} from h^{l}
        return rst
    

class TemporalConvLayer(nn.Module):
    '''
    Section 3.3 in the paper
    Gated 1D temporal convolution layer (Conv2d used in the 1D way due to the input dim)
    ==> We only care about H_out if input:[N, C_in, H_in, W_in] and output:[N, C_out, H_out, W_out]
        where H corresponds to the timestep dimension
    
    Inputs:
        c_in: input channels
        c_out: output channels
        kernel: kernel size for timestep axis
        dia: spacing between kernel elements
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]
            
    Return:
        gated_conv: output with the shape (we assume that dia = 1)
        [batch_size, c_out, timesteps-kernel_size[0]+1, num_nodes-kernel_size[1]+1]
        i.e. [batch, c_out, timestep-1, num_nodes] if kernel_size = (2, 1)
        
    '''
    def __init__(self, c_in, c_out, kernel = 2, dia = 1):
        super(TemporalConvLayer, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, 2*c_out, (kernel, 1), 1, dilation = dia, padding = (0,0))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv_x = self.conv(x)
        P = conv_x[:, 0:self.c_out, :, :]
        Q = conv_x[:, -self.c_out:, :, :]
        gated_conv = P * self.sigmoid(Q)
        return gated_conv


class TemporalConvLayer_Residual(nn.Module):
    '''
    ** 'TemporalConvLayer' with the residual connection **
        
    Inputs:
        c_in: input channels
        c_out: output channels
        kernel: kernel size for timestep axis
        dia: spacing between kernel elements
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]
            
    Return:
        gated_conv: output with the shape (we assume that dia = 1)
        [batch_size, c_out, timesteps-kernel_size[0]+1, num_nodes-kernel_size[1]+1]
        i.e. [batch, c_out, timestep-1, num_nodes] if kernel_size = (2, 1)
        
    '''
    def __init__(self, c_in, c_out, kernel = 2, dia = 1):
        super(TemporalConvLayer_Residual, self).__init__()
        self.c_out = c_out
        self.c_in = c_in
        self.conv = nn.Conv2d(c_in, 2*c_out, (kernel, 1), 1, dilation = dia, padding = (0,0))
        if self.c_in > self.c_out:
            self.conv_self = nn.Conv2d(c_in, c_out, (1, 1))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # get the last two dims of 'x'
        b, _, T, n = list(x.size())
        if self.c_in > self.c_out:
            # [batch, c_out, timesteps, num_nodes]
            x_self = self.conv_self(x)
        elif self.c_in < self.c_out:
            # [batch, c_out, timesteps, num_nodes]
            x_self = torch.cat([x, torch.zeros([b, self.c_out - self.c_in, T, n]).to(x)], dim=1)
        else:
            x_self = x
        conv_x = self.conv(x)
        # get the timesteps dim of 'conv(x)'
        _, _, T_new, _ = list(conv_x.size())
        # need 'x_self' has the same shape of 'P'
        x_self = x_self[:, :, -T_new:, :]
        P = conv_x[:, 0:self.c_out, :, :]
        Q = conv_x[:, -self.c_out:, :, :]
        # residual connection added
        gated_conv = (P + x_self) * self.sigmoid(Q)
        return gated_conv


class SpatialConvLayer(nn.Module):
    '''
    Section 3.2 in the paper
    Graph convolution layer (GCN used here as the spatial CNN)
    
    Inputs:
        c_in: input channels
        c_out: output channels
        g: DGLGraph
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]
        
    Return:
        y: output with the shape [batch_size, c_out, timesteps, num_nodes]
        
    '''
    def __init__(self, c_in, c_out, g):
        super(SpatialConvLayer, self).__init__()
        self.g = g
        self.gc = GCN(c_in, c_out, activation=nn.ReLU())

    def forward(self, x):
        # [batch, c_in, ts, nodes] --> [nodes, c_in, ts, batch]
        x = x.transpose(0, 3)
        # [nodes, c_in, ts, batch] --> [nodes, batch, ts, c_in]
        x = x.transpose(1, 3)
        # output: [nodes, batch, ts, c_out]
        output = self.gc(self.g, x)
        # [nodes, batch, ts, c_out] --> [nodes, c_out, ts, batch]
        output = output.transpose(1, 3)
        # [nodes, c_out, ts, batch] --> [batch, c_out, ts, nodes]
        output = output.transpose(0, 3)
        # return with the shape: [batch, c_out, ts, nodes]
        return torch.relu(output)


class OutputLayer(nn.Module):
    '''
    several temporal conv layers with a fully-connected layer as the output layer
    
    Inputs:
        c: input channels, c_in = c_out = c
        T: same as the timesteps dimention in 'x'
        n: number of nodes
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]
        
    Outputs:
        y: output with the shape [batch_size, 1, 1, num_nodes]
        
    '''
    
    def __init__(self, c, T, n):
        super(OutputLayer, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.ln = nn.LayerNorm([n, c])
        self.tconv2 = nn.Conv2d(c, c, (1, 1), 1, dilation = 1, padding = (0,0))
        self.fc = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        # maps multi-steps to one
        # [batch, c_in, ts, nodes] --> [batch, c_out_1, 1, nodes]
        x_t1 = self.tconv1(x)
        x_ln = self.ln(x_t1.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        # [batch, c_out_1, 1, nodes] --> [batch, c_out_2, 1, nodes]
        x_t2 = self.tconv2(x_ln)
        # maps multi-channels to one
        # [batch, c_out_2, 1, nodes] --> [batch, 1, 1, nodes]
        return self.fc(x_t2)

    
class OutputLayer_Simple(nn.Module):
    '''
    
    ** Simplified version of 'OutputLayer' **
    
    The second half of the section 3.4 in the paper
    Single temporal conv layers with a fully-connected layer as the output layer
    
    Inputs:
        c: input channels, c_in = c_out = c
        T: same as the timesteps dimention in 'x'
        n: number of nodes
        x: input with the shape [batch_size, c_in, timesteps, num_nodes]
        
    Outputs:
        y: output with the shape [batch_size, 1, 1, num_nodes]
        
    '''
    
    def __init__(self, c, T):
        super(OutputLayer_Simple, self).__init__()
        self.tconv1 = nn.Conv2d(c, c, (T, 1), 1, dilation = 1, padding = (0,0))
        self.fc = nn.Conv2d(c, 1, 1)

    def forward(self, x):
        # [batch, c_in, ts, nodes] --> [batch, c_out = c_in, 1, nodes]
        x_t1 = self.tconv1(x)
        # [batch, c_out = c_in, 1, nodes] --> [batch, 1, 1, nodes]
        return self.fc(x_t1)