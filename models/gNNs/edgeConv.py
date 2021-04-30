import dgl
import torch
import torch.nn as nn
from torch_geometric.nn import EdgeConv, knn_graph, DynamicEdgeConv
from dgl.nn.pytorch import GraphConv

from models.gNNs.layers import GNNLayer

class MLP(nn.Module):
    def __init__(self, layer_dims, batch_norm=True, relu=True):
        super().__init__()
        self.layer_dims = layer_dims
        layers = []
        for i in range(len(layer_dims) - 1):
            in_dim, out_dim = layer_dims[i], layer_dims[i + 1]
            layers.append(nn.Linear(in_dim, out_dim))
            if batch_norm:
                layers.append(nn.BatchNorm1d(out_dim))
            if relu:
                layers.append(nn.ReLU())
        self.layer = nn.Sequential(*layers)

    def reset_parameters(self):
        reset(self.layer)

    def forward(self, x):
        return self.layer(x)

class DynEdgeConvGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(DynEdgeConvGCNSegmentation, self).__init__()
        self.edge_conv_dims = [[256, 256, 256], [64, 64, 64], [64, 64]]
        self.edge_convs = self.make_edge_conv_layers_()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, n_classes, activation=None)


    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        dynEdgeConv1 = list(self.edge_convs.children())[0]
        dynEdgeConv2 = list(self.edge_convs.children())[1]
        hidden = self.conv1(graph, features)
        print("in_dim")
        print(3)
        print("hidden_dim")
        print(256)
        print("hidden shape after conv1")
        print(hidden.shape)
        hidden = dynEdgeConv1(hidden)
        print("hidden shape after dynEdgeConv1")
        print(hidden.shape)
        hidden = self.conv2(graph, hidden)
        print("hidden shape after conv2")
        print(hidden.shape)
        hidden = dynEdgeConv2(hidden)
        print("hidden shape after dynEdgeConv2")
        print(hidden.shape)
        return self.conv3(graph, hidden)

    def make_edge_conv_layers_(self):
        """Define structure of the EdgeConv Blocks
        edge_conv_dims: [[convi_mlp_dims]], e.g., [[3, 64], [64, 128]]
        """
        print("Inside make_edge_conv_layers_()")

        layers = []
        for dims in self.edge_conv_dims:
            mlp_dims = [dims[0] * 2] + dims[1::]
            mlp_dims = dims
            print("mlp_dims")
            print(mlp_dims)
            layers.append(DynamicEdgeConv(nn=MLP(mlp_dims), k=20, aggr='max'))
        return nn.Sequential(*layers)
