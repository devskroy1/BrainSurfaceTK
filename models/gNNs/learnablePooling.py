import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial
from torch_geometric.nn import knn_graph, SAGPooling, graclus
from dgl.nn.pytorch import GraphConv

class LearnablePoolingGCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(LearnablePoolingGCNRegressor, self).__init__()
        self.dropout = 0.5
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.Softmax())
        self.conv2 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # print("Inside BasicGCNRegressor forward()")
        # Perform graph convolution and activation function.
        hidden_S1 = self.conv1(graph, features)
        # print("After self.conv1")
        # hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden_Y1 = self.conv2(graph, features)

        g_pool_Y2 = torch.matmul(torch.transpose(hidden_S1, 0, 1), hidden_Y1)
        # hidden = F.dropout(hidden, self.dropout, training=self.training)
        hidden = self.conv3(g_pool_Y2, features)
        # hidden = F.dropout(hidden, self.dropout, training=self.training)
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