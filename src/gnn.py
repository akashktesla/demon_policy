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


import torch
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class EdgePredictor(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(EdgePredictor, self).__init__()
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(in_dim * 2, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, out_dim),
            torch.nn.Sigmoid()
        )

    def forward(self, x, edge_index):
        row, col = edge_index
        edge_input = torch.cat([x[row], x[col]], dim=1)
        return self.mlp(edge_input)

class GraphGenerator(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim_node, hidden_dim_edge, out_dim_node, out_dim_edge):
        super(GraphGenerator, self).__init__()
        self.gnn = MessagePassing(aggr='add', flow='source_to_target')
        self.edge_predictor = EdgePredictor(in_dim, hidden_dim_edge, 1)
        self.node_updater = torch.nn.Sequential(
            torch.nn.Linear(in_dim + 1, hidden_dim_node),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_node, out_dim_node)
        )
        self.edge_attr_generator = torch.nn.Sequential(
            torch.nn.Linear(in_dim * 2, hidden_dim_edge),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim_edge, out_dim_edge)
        )

    def forward(self, x, edge_index, edge_attr):
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.shape[0])
        row, col = edge_index
        deg = degree(row, x.shape[0], dtype=torch.float)
        norm = deg.pow(-0.5)
        norm[torch.isinf(norm)] = 0
        x = x * norm.view(-1, 1)

        # Message passing
        x = self.gnn(x, edge_index)

        # Edge prediction
        edge_prob = self.edge_predictor(x, edge_index)

        # Edge sampling
        new_edge_index = torch.topk(edge_prob, k=int(edge_prob.shape[1] / 2), dim=-1)[1]

        # Node feature update
        new_x = self.node_updater(torch.cat([x, edge_prob], dim=1))

        # Edge attribute generation
        new_edge_attr = self.edge_attr_generator(torch.cat([new_x[row], new_x[col]], dim=1))

        return new_x, new_edge_index, new_edge_attr



