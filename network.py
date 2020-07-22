# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 18:30:55 2020

@author: Ming Jin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
from dgl import DGLGraph
from layers import TemporalConvLayer, TemporalConvLayer_Residual, SpatialConvLayer, OutputLayer, OutputLayer_Simple


class STGCN(nn.Module):
    '''
    STGCN network described in the paper (Figure 2)
    
    Inputs:
        c: channels, e.g. [1, 64, 16, 64, 64, 16, 64] where [1, 64] means c_in and c_out for the first temporal layer
        T: window length, e.g. 12
        n: num_nodes
        g: fixed DGLGraph
        p: dropout after each 'sandwich', i.e. 'TSTN', block
        control_str: model strcture controller, e.g. 'TSTNTSTN'; T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]
        
    Return:
        y: output with the shape [batch, 1, 1, n]
        
    '''

    def __init__(self, c, T, n, g, p, control_str):
        
        super(STGCN, self).__init__()
        
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.num_nodes = n
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        # Temporal conv kernel size set to 3
        self.Kt = 3
        # c_index controls the change of channels
        c_index = 0
        num_temporal_layers = 0
        
        # construct network based on 'control_str'
        for i in range(self.num_layers):
            
            layer_i = control_str[i]
            
            # Temporal Layer
            if layer_i == 'T':
                self.layers.append(TemporalConvLayer_Residual(c[c_index], c[c_index + 1], kernel = self.Kt))
                c_index += 1
                num_temporal_layers += 1
                
            # Spatio Layer
            if layer_i == 'S':
                self.layers.append(SpatialConvLayer(c[c_index], c[c_index + 1], g))
                c_index += 1
                
            # Norm Layer
            if layer_i == 'N':
                # TODO: The meaning of this layernorm
                self.layers.append(nn.LayerNorm([n, c[c_index]]))
        
        # c[c_index] is the last element in 'c'
        # T - (self.Kt - 1) * num_temporal_layers returns the timesteps after previous temporal layer transformations cuz dialiation = 1
        self.output = OutputLayer(c[c_index], T - (self.Kt - 1) * num_temporal_layers, self.num_nodes)
        
        for layer in self.layers:
            layer = layer.cuda()
            
    def forward(self, x):
        # Example:
        # batch=64, input_channel=1, window_length=12, num_nodes=207, temporal_kernel = 2
        #   input.shape: torch.Size([64, 1, 12, 207])
        #   T output.shape: torch.Size([64, 64, 11, 207])
        #   S output.shape: torch.Size([64, 16, 11, 207])
        #   T output.shape: torch.Size([64, 64, 10, 207])
        #   N output.shape: torch.Size([64, 64, 10, 207])
        #   T output.shape: torch.Size([64, 64, 9, 207])
        #   S output.shape: torch.Size([64, 16, 9, 207])
        #   T output.shape: torch.Size([64, 64, 8, 207])
        #   OutputLayer output.shape: torch.Size([64, 1, 1, 207])

        for i in range(self.num_layers):
            layer_i = self.control_str[i]
            if layer_i == 'N':
                # x.permute(0, 2, 3, 1) leads
                # [batch, channel, timesteps, nodes] to [batch, timesteps, nodes, channel]
                # self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) leads
                # [batch, timesteps, nodes, channel] to [batch, channel, timesteps, nodes]
                x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))
            else:
                # x.shape is [batch, channel, timesteps, nodes]
                x = self.layers[i](x)
        return self.output(x)  # [batch, 1, 1, nodes]



class STGCN_WAVE(nn.Module):
    '''
    Improved variation of the above STGCN network
        + Extra temporal conv, layernorm, and sophisticated output layer design
        + temporal conv with increasing dialations like in TCN
    
    Inputs:
        c: channels, e.g. [1, 16, 32, 64, 32, 128] where [1, 16] means c_in and c_out for the first temporal layer
        T: window length, e.g. 144, which should larger than the total dialations
        n: num_nodes
        g: fixed DGLGraph
        p: dropout
        control_str: model strcture controller, e.g. 'TNTSTNTSTN'; T: Temporal Layer, S: Spatio Layer, N: Norm Layer
        x: input feature matrix with the shape [batch, 1, T, n]
        
    Return:
        y: output with the shape [batch, 1, 1, n]
        
    Notice:
        ** Temporal layer changes c_in to c_out, but spatial layer doesn't change
           in this way where c_in = c_out = c
    '''

    def __init__(self, c, T, n, g, p, control_str):
        
        super(STGCN_WAVE, self).__init__()
        
        self.control_str = control_str
        self.num_layers = len(control_str)
        self.layers = nn.ModuleList()
        self.dropout = nn.Dropout(p)
        # c_index controls the change of channels
        c_index = 0
        # diapower controls the change of dilations in temporal CNNs where dilation = 2^diapower
        diapower = 0
        
        # construct network based on 'control_str'
        for i in range(self.num_layers):
            
            layer_i = control_str[i]
            
            # Temporal Layer
            if layer_i == 'T':
                # Notice: dialation = 2^diapower (e.g. 1, 2, 4, 8) so that 
                # T_out = T_in - dialation * (kernel_size - 1) - 1 + 1
                # if padding = 0 and stride = 1
                self.layers.append(TemporalConvLayer_Residual(c[c_index], c[c_index + 1], dia = 2**diapower))
                diapower += 1
                c_index += 1
                
            # Spatio Layer
            if layer_i == 'S':
                self.layers.append(SpatialConvLayer(c[c_index], c[c_index], g))
                
            # Norm Layer
            if layer_i == 'N':
                # TODO: The meaning of this layernorm
                self.layers.append(nn.LayerNorm([n, c[c_index]]))
        
        # c[c_index] is the last element in 'c'
        # T + 1 - 2**(diapower) returns the timesteps after previous temporal layer transformations
        # 'n' will be needed by LayerNorm inside of the OutputLayer
        self.output = OutputLayer(c[c_index], T + 1 - 2**(diapower), n)
        
        for layer in self.layers:
            layer = layer.cuda()
            
    def forward(self, x):
        # Example:
        # batch=8, input_channel=1, window_length=144, num_nodes=207, temporal_kernel = 2
        #   x.shape: torch.Size([8, 1, 144, 207])
        #   T output.shape: torch.Size([8, 16, 143, 207])
        #   N output.shape: torch.Size([8, 16, 143, 207])
        #   T output.shape: torch.Size([8, 32, 141, 207])
        #   S output.shape: torch.Size([8, 32, 141, 207])
        #   T output.shape: torch.Size([8, 64, 137, 207])
        #   N output.shape: torch.Size([8, 64, 137, 207])
        #   T output.shape: torch.Size([8, 32, 129, 207])
        #   S output.shape: torch.Size([8, 32, 129, 207])
        #   T output.shape: torch.Size([8, 128, 113, 207])
        #   Outputlayer output.shape: torch.Size([8, 1, 1, 207]) 
           
        for i in range(self.num_layers):
            layer_i = self.control_str[i]
            if layer_i == 'N':
                # x.permute(0, 2, 3, 1) leads
                # [batch, channel, timesteps, nodes] to [batch, timesteps, nodes, channel]
                # self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) leads
                # [batch, timesteps, nodes, channel] to [batch, channel, timesteps, nodes]
                x = self.dropout(self.layers[i](x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2))  
            else:
                # x.shape is [batch, channel, timesteps, nodes]
                x = self.layers[i](x)
        return self.output(x)  # [batch, 1, 1, nodes]