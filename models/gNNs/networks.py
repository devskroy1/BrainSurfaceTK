import dgl
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from torch_geometric.nn import global_mean_pool
from models.gNNs.layers import GNNLayer

#Add more convolutional layers, and linear layers. Play with hyperparams, like LR. Watch both training and val accuracy.
#Watch for overfitting on training data
# Play with num layers, hidden dims (size of feature maps), fc layers.
# How deep is network, and how big it is (num feature maps)
#Compare first few epochs. Narrow search from first few epochs. Plot losses, see which curves converge faster.
#Start with big LRs, e.g. 0.1, then reduce LR. Be systematic
class BasicGCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(BasicGCNRegressor, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)
        self.device = device

    def forward(self, graph, features, is_training):
        # Perform graph convolution and activation function.

        # print("Inside GCNRegressor forward()")
        # print("graph")
        # print(graph)
        # print("Features shape")
        # print(features.shape)
        # total_num_nodes = features.size(0)
        # print("total_num_nodes")
        # print(total_num_nodes)
        # num_nodes_per_graph = total_num_nodes // self.batch_size

        # print("num_nodes_per_graph")
        # print(num_nodes_per_graph)

        hidden = self.conv1(graph, features)
        # print("hidden shape")
        # print(hidden.shape)
        hidden = self.conv2(graph, hidden)

        # print("conv2 hidden shape")
        # print(hidden.shape)
        # batch_cat_tensor = torch.zeros(num_nodes_per_graph, dtype=torch.int64, device=self.device)
        # for b in range(1, self.batch_size):
        #     batch_tensor = torch.ones(num_nodes_per_graph, dtype=torch.int64, device=self.device) * b
        #     # batch_tensor = batch_tensor.new_full((, npoint), b)
        #     # print("batch_tensor shape")
        #     # print(batch_tensor.shape)
        #     # Expected to be 512
        #     batch_cat_tensor = torch.cat([batch_cat_tensor, batch_tensor], dim=0)
        #
        # num_nodes_multiple = batch_cat_tensor.size(0)
        # remainder_batch_tensor = torch.ones(total_num_nodes - num_nodes_multiple, dtype=torch.int64, device=self.device) * (self.batch_size - 1)
        # batch_cat_tensor = torch.cat([batch_cat_tensor, remainder_batch_tensor], dim=0)

        # print("batch_cat_tensor shape")
        # print(batch_cat_tensor.shape)
        #Global Average Pooling on output of final conv layer
        #gap_output = global_mean_pool(hidden, batch_cat_tensor, self.batch_size)
        #gap_output = global_mean_pool(hidden)

        gap_output = torch.mean(hidden.view(hidden.size(0), hidden.size(1), -1), dim=2)

        # print("gap_output shape")
        # print(gap_output.shape)
        with graph.local_scope():
            #graph.ndata['tmp'] = hidden
            graph.ndata['tmp'] = gap_output
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        seg_output = self.predict_layer(hg)

        if is_training:
            return seg_output

        #For populn level saliency map
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
            # print("class_activn_map shape")
            # print(class_activn_map.shape)
            graph.ndata['saliency_score'] = class_activn_map[0]
            # print("graph.ndata['saliency_score']")
            # print(graph.ndata['saliency_score'])
            # print("graph")
            # print(graph)
            # print("Just before returning from forward()")
            return graph, seg_output

        # if is_training:
        #     linear_weights = self.predict_layer.weight
        #     # print("linear_weights shape")
        #     # print(linear_weights.shape)
        #     n_classes = linear_weights.size(0)
        #     hidden_dim = linear_weights.size(1)
        #     hidden_first_dim = hidden.size(0)
        #     class_activn_map = torch.zeros(size=(n_classes, hidden_first_dim), device=self.device)
        #
        #     for h in range(hidden_dim):
        #         weight = linear_weights[:, h]
        #         # weight = weight.to(self.device)
        #         # print("weight shape")
        #         # print(weight.shape)
        #         conv_feature_map = hidden[:, h]
        #         # conv_feature_map = conv_feature_map.to(self.device)
        #         # print("conv_feature_map shape")
        #         # print(conv_feature_map.shape)
        #         class_activn_map += torch.matmul(weight.unsqueeze(1), conv_feature_map.unsqueeze(0))
        #     # print("class_activn_map shape")
        #     # print(class_activn_map.shape)
        #     graph.ndata['saliency_score'] = class_activn_map[0]
        #     # print("graph.ndata['saliency_score']")
        #     # print(graph.ndata['saliency_score'])
        #     # print("graph")
        #     # print(graph)
        #     # print("Just before returning from forward()")
        #     return graph, seg_output

        # For basic saliency scores
        # else:
        #     linear_weights = self.predict_layer.weight
        #     # print("linear_weights shape")
        #     # print(linear_weights.shape)
        #     n_classes = linear_weights.size(0)
        #     hidden_dim = linear_weights.size(1)
        #     hidden_first_dim = hidden.size(0)
        #     class_activn_map = torch.zeros(size=(n_classes, hidden_first_dim), device=self.device)
        #
        #     for h in range(hidden_dim):
        #         weight = linear_weights[:, h]
        #         # weight = weight.to(self.device)
        #         # print("weight shape")
        #         # print(weight.shape)
        #         conv_feature_map = hidden[:, h]
        #         # conv_feature_map = conv_feature_map.to(self.device)
        #         # print("conv_feature_map shape")
        #         # print(conv_feature_map.shape)
        #         class_activn_map += torch.matmul(weight.unsqueeze(1), conv_feature_map.unsqueeze(0))
        #     # print("class_activn_map shape")
        #     # print(class_activn_map.shape)
        #     # graph.ndata['saliency_score'] = class_activn_map[0]
        #     # print("graph.ndata['saliency_score']")
        #     # print(graph.ndata['saliency_score'])
        #     # print("graph")
        #     # print(graph)
        #     # print("Just before returning from forward()")
        #     return seg_output, class_activn_map

class BasicGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(BasicGCNSegmentation, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, n_classes, activation=None)

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = self.conv2(graph, hidden)
        return self.conv3(graph, hidden)

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
