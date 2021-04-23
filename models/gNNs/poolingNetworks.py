import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import global_sort_pool, global_mean_pool, EdgePooling
from dgl.nn.pytorch import GraphConv

from models.gNNs.layers import GNNLayer

class PooledGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, batch_size, device):
        super(PooledGCNSegmentation, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, n_classes, activation=None)
        self.batch_size = batch_size
        self.device = device

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        print("graph")
        print(graph)
        hidden = self.conv1(graph, features)
        hidden = self.conv2(graph, hidden)
        print("conv2 output shape")
        print("hidden shape")
        print(hidden.shape)
        k = 5000
        N = hidden.size(0)
        batch_cat_tensor = torch.zeros(N // self.batch_size, dtype=torch.int64, device=self.device)
        for b in range(1, self.batch_size):
            batch_tensor = torch.ones(N // self.batch_size, dtype=torch.int64, device=self.device) * b
            # batch_tensor = batch_tensor.new_full((, npoint), b)
            # print("batch_tensor shape")
            # print(batch_tensor.shape)
            # Expected to be 512
            batch_cat_tensor = torch.cat([batch_cat_tensor, batch_tensor], dim=0)
        len_batch_vector = batch_cat_tensor.size(0)
        len_difference = N - len_batch_vector
        batch_tensor = torch.ones(len_difference, dtype=torch.int64, device=self.device) * (self.batch_size - 1)
        batch_cat_tensor = torch.cat([batch_cat_tensor, batch_tensor], dim=0)

        hidden = global_sort_pool(hidden, batch_cat_tensor, k)
        print("global_sort_pool layer output shape")
        print(hidden.shape)

        return self.conv3(graph, hidden)
