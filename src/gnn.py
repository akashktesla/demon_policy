import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import torch.relu


class GNNModel_base(nn.module):
    def __init__(self,n_input,n_hidden,n_output):
        super(GNNModel,self).__init__()
        self.conv1 = GCNConv(n_input,n_hidden) 
        self.conv2 = GCNConv(n_hidden,n_output) 

    def forward(x,edge_index,edge_attr):
        x = self.conv1(x,edge_index,edge_attr)
        x = relu(x)
        x = self.conv2(x,edge_index,edge_attr)
        x = relu(x)
        return x






