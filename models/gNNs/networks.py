import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv

from models.gNNs.layers import GNNLayer


class BasicGCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(BasicGCNRegressor, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = self.conv2(graph, hidden)
        with graph.local_scope():
            #graph.ndata['tmp'] = hidden
            graph.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        return self.predict_layer(hg)

class BasicGCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(BasicGCNClassifier, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = self.conv2(graph, hidden)
        with graph.local_scope():
            #graph.ndata['tmp'] = hidden
            graph.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        return self.predict_layer(hg)

class BasicGCNSegmentation(nn.Module):
    #TODO: Remove linear layer, CAM Code, with graph local scope code in forward, prints
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(BasicGCNSegmentation, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)
        self.device = device

    def forward(self, graph, features, is_training=True):
        # Perform graph convolution and activation function.
        print("Inside BasicGCNSegmentation forward()")
        print("graph")
        print(graph)
        print("features shape")
        print(features.shape)
        hidden = self.conv1(graph, features)
        print("hidden conv1 shape")
        print(hidden.shape)
        hidden = self.conv2(graph, hidden)
        print("hidden conv2 shape")
        print(hidden.shape)
        hidden = self.conv3(graph, hidden)
        print("hidden conv3 shape")
        print(hidden.shape)

        # print("final conv layer output feature map shape")
        # print(hidden.shape)

        gap_output = torch.mean(hidden.view(hidden.size(0), hidden.size(1), -1), dim=2)

        print("gap_output shape")
        print(gap_output.shape)

        with graph.local_scope():
            #graph.ndata['tmp'] = hidden
            graph.ndata['tmp'] = gap_output
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        seg_output = self.predict_layer(hg)

        print("seg_output shape")
        print(seg_output.shape)
        #Should be [1275220, 40] not [64, 40]

        if is_training:
            #Todo: remove this
            linear_weights = self.predict_layer.weight
            # print("linear_weights shape")
            # print(linear_weights.shape)
            n_classes = linear_weights.size(0)
            hidden_dim = linear_weights.size(1)
            hidden_first_dim = hidden.size(0)
            class_activn_map = torch.zeros(size=(n_classes, hidden_first_dim), device=self.device)

            for h in range(hidden_dim):
                weight = linear_weights[:, h]
                # weight = weight.to(self.device)
                # print("weight shape")
                # print(weight.shape)
                conv_feature_map = hidden[:, h]
                # conv_feature_map = conv_feature_map.to(self.device)
                # print("conv_feature_map shape")
                # print(conv_feature_map.shape)
                class_activn_map += torch.matmul(weight.unsqueeze(1), conv_feature_map.unsqueeze(0))

            # print("class_activn_map shape")
            # print(class_activn_map.shape)
            # print(class_activn_map)
            #

            return seg_output

        else:
            linear_weights = self.predict_layer.weight
            # print("linear_weights shape")
            # print(linear_weights.shape)
            n_classes = linear_weights.size(0)
            hidden_dim = linear_weights.size(1)
            hidden_first_dim = hidden.size(0)
            class_activn_map = torch.zeros(size=(n_classes, hidden_first_dim), device=self.device)

            for h in range(hidden_dim):
                weight = linear_weights[:, h]
                # weight = weight.to(self.device)
                # print("weight shape")
                # print(weight.shape)
                conv_feature_map = hidden[:, h]
                # conv_feature_map = conv_feature_map.to(self.device)
                # print("conv_feature_map shape")
                # print(conv_feature_map.shape)
                class_activn_map += torch.matmul(weight.unsqueeze(1), conv_feature_map.unsqueeze(0))

            return seg_output, class_activn_map

        # return self.conv3(graph, hidden)


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
