import dgl
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import knn_graph
from dgl.nn.pytorch import GraphConv

from models.gNNs.layers import GNNLayer


class BasicGCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(BasicGCNRegressor, self).__init__()
        self.dropout = 0.5
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # print("Inside BasicGCNRegressor forward()")
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        # print("After self.conv1")
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv2(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv3(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv4(graph, hidden)
        # hidden = F.dropout(hidden, self.dropout, training=self.training)
        # hidden = self.conv5(graph, hidden)
        # hidden = F.dropout(hidden, self.dropout, training=self.training)
        # hidden = self.conv6(graph, hidden)
        with graph.local_scope():
            graph.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        return self.predict_layer(hg)


class BasicGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(BasicGCNSegmentation, self).__init__()
        self.dropout = 0.5
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv5 = GraphConv(hidden_dim, n_classes, activation=None)


    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = knn_graph(hidden, 20)
        print("out shape")
        print(out.shape)
        hidden = self.conv2(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv3(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv4(graph, hidden)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        return self.conv5(graph, hidden)

class GNNModel(nn.Module):
    def __init__(self, input_node_dim, input_edge_dim, hidden_dim1, hidden_dim2, out_dim):
        super(GNNModel, self).__init__()
        self.gn1 = GNNLayer(input_node_dim, input_edge_dim, hidden_dim1, activation=nn.ReLU())
        self.gc1 = GraphConv(hidden_dim1, hidden_dim2, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim2, out_dim)

    def forward(self, g, node_features, edge_features):
        hidden = self.gn1(g, node_features, edge_features)
        hidden = self.gc1(g, hidden)
        with g.local_scope():
            g.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(g, 'tmp')
        return self.predict_layer(hg)
