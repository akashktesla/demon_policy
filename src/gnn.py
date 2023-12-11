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


class WordEncoder(torch.nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(WordEncoder, self).__init__()
        self.embedding = Embedding(vocab_size, embedding_dim)
        self.linear1 = Linear(embedding_dim * len(self.embedding.weight), hidden_dim)
        self.linear2 = Linear(hidden_dim, hidden_dim)
        self.linear3 = Linear(hidden_dim, output_size)

    def forward(self, word_list):
        embedded_words = self.embedding(word_list)
        flattened_embedding = torch.flatten(embedded_words, 1)
        hidden1 = torch.relu(self.linear1(flattened_embedding))
        hidden2 = torch.relu(self.linear2(hidden1))
        encoded_list = self.linear3(hidden2)
        return encoded_list
