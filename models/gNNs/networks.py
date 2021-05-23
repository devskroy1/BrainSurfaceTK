import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import scipy.spatial
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import knn_graph, SAGPooling, graclus, EdgePooling
from dgl.nn.pytorch import GraphConv, SortPooling, MaxPooling, AvgPooling, GlobalAttentionPooling
from dgl.subgraph import node_subgraph

from models.gNNs.layers import GNNLayer

def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])

#Basic GCN Regressor
class BasicGCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(BasicGCNRegressor, self).__init__()
        #self.dropout = 0.5
        self.hidden_dim = hidden_dim
        self.device = device
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
       # self.globalAttentionPooling1 = GlobalAttentionPooling(MLP([hidden_dim, hidden_dim, hidden_dim, 1]))

        #self.globalAttentionPooling2 = GlobalAttentionPooling(MLP([hidden_dim*2, hidden_dim*2, hidden_dim*2, hidden_dim*4]))

        # self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv7 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        # self.conv8 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.predict_layer = nn.Linear(hidden_dim, n_classes)

    def forward(self, graph, features):
        # Perform graph convolution and activation function.
        hidden = self.conv1(graph, features)
        hidden = self.conv2(graph, hidden)
        hidden = self.conv3(graph, hidden)
        hidden = self.conv4(graph, hidden)

        with graph.local_scope():
            graph.ndata['tmp'] = hidden
            # Calculate graph representation by averaging all the node representations.
            hg = dgl.mean_nodes(graph, 'tmp')

        # print("hg shape")
        # print(hg.shape)
        # return self.predict_layer(hg)

        # print("hg shape")
        # print(hg.shape)
        # global_attention_pooling_feats = self.globalAttentionPooling1(graph, hidden)
        # # print("global_attention_pooling_feats shape")
        # # print(global_attention_pooling_feats.shape)
        # global_attention_pooling_hg = torch.cat((hg, global_attention_pooling_feats), dim=1)
        # # print("max_pool_hg shape torch cat")
        # # print(max_pool_hg.shape)
        #
        # # max_pool_hg = self.max_pool1(graph, hg)
        # # print("global_attention_pooling_hg shape")
        # # print(global_attention_pooling_hg.shape)
        #
        # return self.predict_layer(global_attention_pooling_hg)
        return self.predict_layer(hg)

# #GCN Regressor with GlobalAttentionPooling
# class BasicGCNRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, device):
#         super(BasicGCNRegressor, self).__init__()
#         self.dropout = 0.5
#         self.hidden_dim = hidden_dim
#         self.device = device
#         self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
#         self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.globalAttentionPooling1 = GlobalAttentionPooling(MLP([hidden_dim, hidden_dim, hidden_dim, 1]))
#
#         #self.globalAttentionPooling2 = GlobalAttentionPooling(MLP([hidden_dim*2, hidden_dim*2, hidden_dim*2, hidden_dim*4]))
#
#         # self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv7 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv8 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.predict_layer = nn.Linear(hidden_dim * 2, n_classes)
#
#     def forward(self, graph, features):
#         # Perform graph convolution and activation function.
#         hidden = self.conv1(graph, features)
#         hidden = self.conv2(graph, hidden)
#         hidden = self.conv3(graph, hidden)
#         hidden = self.conv4(graph, hidden)
#
#         with graph.local_scope():
#             graph.ndata['tmp'] = hidden
#             # Calculate graph representation by averaging all the node representations.
#             hg = dgl.mean_nodes(graph, 'tmp')
#
#         # print("hg shape")
#         # print(hg.shape)
#         # return self.predict_layer(hg)
#
#         # print("hg shape")
#         # print(hg.shape)
#         global_attention_pooling_feats = self.globalAttentionPooling1(graph, hidden)
#         # print("global_attention_pooling_feats shape")
#         # print(global_attention_pooling_feats.shape)
#         global_attention_pooling_hg = torch.cat((hg, global_attention_pooling_feats), dim=1)
#         # print("max_pool_hg shape torch cat")
#         # print(max_pool_hg.shape)
#
#         # max_pool_hg = self.max_pool1(graph, hg)
#         # print("global_attention_pooling_hg shape")
#         # print(global_attention_pooling_hg.shape)
#
#         return self.predict_layer(global_attention_pooling_hg)
#         #return self.predict_layer(hg)

#GCN Regressor with Max/Avg pooling
# class BasicGCNRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, device):
#         super(BasicGCNRegressor, self).__init__()
#         self.dropout = 0.5
#         self.hidden_dim = hidden_dim
#         self.device = device
#         self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
#         self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.sort_pool1 = SortPooling(k=1000)
#         # self.sort_pool2 = SortPooling(k=100)
#         self.max_pool1 = MaxPooling()
#         self.max_pool2 = MaxPooling()
#         self.avg_pool1 = AvgPooling()
#         # self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv7 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv8 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.predict_layer = nn.Linear(hidden_dim * 2, n_classes)
#
#     def forward(self, graph, features):
#         # print("Inside BasicGCNRegressor forward()")
#         # Perform graph convolution and activation function.
#         hidden = self.conv1(graph, features)
#         # print("After self.conv1")
#         #hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # max_pool_1_out = self.max_pool1(graph, hidden)
#         #
#         # print("max_pool_1_out shape")
#         # print(max_pool_1_out.shape)
#
#         hidden = self.conv2(graph, hidden)
#
#         hidden = self.conv3(graph, hidden)
#         hidden = self.conv4(graph, hidden)
#         # print("hidden.shape")
#         # print(hidden.shape)
#
#         # topk_node_idxs_batches, sort_pool_out_1 = self.sort_pool1(graph, hidden)
#         # graphs = dgl.unbatch(graph)
#         # batch_size = len(graphs)
#         # subgraphs = []
#         # for g in range(batch_size):
#         #     topk_node_idxs = topk_node_idxs_batches[g]
#         #     subgraph = node_subgraph(graphs[g], topk_node_idxs)
#         #     subgraphs.append(subgraph)
#         # batched_subgraph = dgl.batch(subgraphs)
#
#         #sort_pool_out_1 = sort_pool_out_1.reshape(batch_size * self.sort_pool1.k, -1)
#
#         # hidden = self.conv2(batched_subgraph, sort_pool_out_1)
#         # print("conv2 out shape")
#         # print(hidden.shape)
#
#        # hidden = F.dropout(hidden, self.dropout, training=self.training)
#        #  topk_node_idxs_batches, sort_pool_out_2 = self.sort_pool2(batched_subgraph, hidden)
#        #  graphs = dgl.unbatch(batched_subgraph)
#        #  subgraphs = []
#        #  for g in range(batch_size):
#        #      topk_node_idxs = topk_node_idxs_batches[g]
#        #      subgraph = node_subgraph(graphs[g], topk_node_idxs)
#        #      subgraphs.append(subgraph)
#        #  batched_subgraph = dgl.batch(subgraphs)
#        #
#        #  # x, edge_index, batch, unpool_info = edgePooling(hidden.to(self.device), edge_index.to(self.device), batch.to(self.device))
#        #  # x, edge_index, batch, unpool_info = edgePooling(hidden, edge_index, batch)
#        #
#        #  # hidden = self.conv3(graph, x)
#        #
#        #  sort_pool_out_2 = sort_pool_out_2.reshape(batch_size * self.sort_pool2.k, -1)
#        #  hidden = self.conv3(batched_subgraph, sort_pool_out_2)
#        #  # print("conv3 out shape")
#        #  # print(hidden.shape)
#        # # hidden = F.dropout(hidden, self.dropout, training=self.training)
#        #  hidden = self.conv4(batched_subgraph, hidden)
#         # print("conv4 out shape")
#         # print(hidden.shape)
#
#         # hidden = self.conv2(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv3(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv4(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv5(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv6(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv7(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv8(graph, hidden)
#
#         #max_pool_feats = self.max_pool1(graph, hidden)
#         avg_pool_feats = self.avg_pool1(graph, hidden)
#         # print("max_pool_feats shape")
#         # print(max_pool_feats.shape)
#         with graph.local_scope():
#             graph.ndata['tmp'] = hidden
#             # Calculate graph representation by averaging all the node representations.
#             hg = dgl.mean_nodes(graph, 'tmp')
#
#         # print("hg shape")
#         # print(hg.shape)
#         avg_pool_hg = torch.cat((hg, avg_pool_feats), dim=1)
#         # print("max_pool_hg shape torch cat")
#         # print(max_pool_hg.shape)
#
#         # max_pool_hg = self.max_pool1(graph, hg)
#         # print("max_pool_hg shape max pool")
#         # print(max_pool_hg.shape)
#
#         return self.predict_layer(avg_pool_hg)
#         #return self.predict_layer(hg)

#GCN Regressor with Sort pooling
# class BasicGCNRegressor(nn.Module):
#     def __init__(self, in_dim, hidden_dim, n_classes, device):
#         super(BasicGCNRegressor, self).__init__()
#         self.dropout = 0.5
#         self.hidden_dim = hidden_dim
#         self.device = device
#         self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
#         self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.conv4 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.sort_pool1 = SortPooling(k=1000)
#         self.sort_pool2 = SortPooling(k=100)
#         self.max_pool1 = MaxPooling()
#         self.max_pool2 = MaxPooling()
#         # self.conv5 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv6 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv7 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         # self.conv8 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
#         self.predict_layer = nn.Linear(hidden_dim, n_classes)
#
#     def forward(self, graph, features):
#         # print("Inside BasicGCNRegressor forward()")
#         # Perform graph convolution and activation function.
#         hidden = self.conv1(graph, features)
#         # print("After self.conv1")
#         #hidden = F.dropout(hidden, self.dropout, training=self.training)
#
#         topk_node_idxs_batches, sort_pool_out_1 = self.sort_pool1(graph, hidden)
#         graphs = dgl.unbatch(graph)
#         batch_size = len(graphs)
#         subgraphs = []
#         for g in range(batch_size):
#             topk_node_idxs = topk_node_idxs_batches[g]
#             subgraph = node_subgraph(graphs[g], topk_node_idxs)
#             subgraphs.append(subgraph)
#         batched_subgraph = dgl.batch(subgraphs)
#
#         sort_pool_out_1 = sort_pool_out_1.reshape(batch_size * self.sort_pool1.k, -1)
#         hidden = self.conv2(batched_subgraph, sort_pool_out_1)
#         # print("conv2 out shape")
#         # print(hidden.shape)
#
#        # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         topk_node_idxs_batches, sort_pool_out_2 = self.sort_pool2(batched_subgraph, hidden)
#         graphs = dgl.unbatch(batched_subgraph)
#         subgraphs = []
#         for g in range(batch_size):
#             topk_node_idxs = topk_node_idxs_batches[g]
#             subgraph = node_subgraph(graphs[g], topk_node_idxs)
#             subgraphs.append(subgraph)
#         batched_subgraph = dgl.batch(subgraphs)
#
#         # x, edge_index, batch, unpool_info = edgePooling(hidden.to(self.device), edge_index.to(self.device), batch.to(self.device))
#         # x, edge_index, batch, unpool_info = edgePooling(hidden, edge_index, batch)
#
#         # hidden = self.conv3(graph, x)
#
#         sort_pool_out_2 = sort_pool_out_2.reshape(batch_size * self.sort_pool2.k, -1)
#         hidden = self.conv3(batched_subgraph, sort_pool_out_2)
#         # print("conv3 out shape")
#         # print(hidden.shape)
#        # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         hidden = self.conv4(batched_subgraph, hidden)
#         # print("conv4 out shape")
#         # print(hidden.shape)
#
#         # hidden = self.conv2(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv3(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv4(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv5(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv6(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv7(graph, hidden)
#         # hidden = F.dropout(hidden, self.dropout, training=self.training)
#         # hidden = self.conv8(graph, hidden)
#         with batched_subgraph.local_scope():
#             batched_subgraph.ndata['tmp'] = hidden
#             # Calculate graph representation by averaging all the node representations.
#             hg = dgl.mean_nodes(batched_subgraph, 'tmp')
#
#         return self.predict_layer(hg)

class BasicGCNSegmentation(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, device):
        super(BasicGCNSegmentation, self).__init__()
        self.dropout = 0.5
        self.hidden_dim = hidden_dim
        self.conv1 = GraphConv(in_dim, hidden_dim, activation=nn.ReLU())
        self.conv2 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv3 = GraphConv(hidden_dim, hidden_dim, activation=nn.ReLU())
        self.conv4 = GraphConv(hidden_dim, n_classes, activation=nn.ReLU())
        self.sort_pool1 = SortPooling(k=1000)
        self.sort_pool2 = SortPooling(k=100)
        #self.conv5 = GraphConv(hidden_dim, n_classes, activation=None)
        # self.sagPooling1 = SAGPooling(hidden_dim)
        # self.sagPooling2 = SAGPooling(hidden_dim)
        self.device = device

    def forward(self, graph, features):
        # Perform graph convolution and activation function.

        hidden = self.conv1(graph, features)
        hidden = F.dropout(hidden, self.dropout, training=self.training)

        # edge_index = self.knn(hidden, hidden, 20)
        #
        # # print("edge_index dtype")
        # # print(edge_index.dtype)
        # edgePooling = EdgePooling(in_channels=self.hidden_dim).to(self.device)
        # graphs = dgl.unbatch(graph)
        #
        # batch_vector = torch.empty(graph.num_nodes(), device=self.device)
        # start_index = 0
        # for g in range(len(graphs)):
        #     num_nodes_graph = graphs[g].num_nodes()
        #     batch_vector[start_index : start_index + num_nodes_graph - 1] = g
        #     start_index += num_nodes_graph
        #
        # # graph = graph.to(self.device)
        # # features = features.to(self.device)
        # batch_vector = batch_vector.long()
        # x, edge_index, batch, unpool_info = edgePooling(hidden.to(self.device), edge_index.to(self.device), batch_vector)



        #x, edge_index, batch, unpool_info = edgePooling(hidden, edge_index, batch_vector)

        # print("hidden shape")
        # print(hidden.shape)
        # print("edge_index shape")
        # print(edge_index.shape)

        #hidden = graclus(edge_index)

        # sag_pool_out1 = self.sagPooling1(hidden.to(self.device), edge_index.to(self.device))
        # # print("sag_pool_out1[0] shape")
        # # print(sag_pool_out1[0].shape)
        # # print("graph")
        # # print(graph)
        # sag_pooled_features = sag_pool_out1[0]
        # sag_edge_index = sag_pool_out1[1]
        # print("graph")
        # print(graph)
        # print("sag_pooled_features shape")
        # print(sag_pooled_features.shape)
        # print("sag_edge_index shape before flattening")
        # print(sag_edge_index.shape)
        # sag_edge_index = torch.flatten(sag_edge_index)
        # print("flattened sag_edge_index shape")
        # print(sag_edge_index.shape)
        # graph = edge_subgraph(graph, sag_edge_index, True)
        # print("subgraph1")
        # print(graph)
        # graph = dgl.add_self_loop(graph)
        # hidden = self.conv2(graph, sag_pooled_features)
        # print("graclus hidden shape")
        # print(hidden.shape)
        # print("graph")
        # print(graph)

        #hidden = self.conv2(graph.to(self.device), hidden.to(self.device))
        print("Before conv2")
        print("graph ")
        print(graph)
        print("hidden shape")
        print(hidden.shape)
        # print("x shape")
        # print(x.shape)
        topk_node_idxs_batches, sort_pool_out_1 = self.sort_pool1(graph, hidden)
        graphs = dgl.unbatch(graph)
        batch_size = len(graphs)
        subgraphs = []
        for g in range(batch_size):
            topk_node_idxs = topk_node_idxs_batches[g]
            subgraph = node_subgraph(graphs[g], topk_node_idxs)
            subgraphs.append(subgraph)
        batched_subgraph = dgl.batch(subgraphs)

        print("topk_node_idxs_batches shape")
        print(topk_node_idxs_batches.shape)
        #Should be 2 X 1000
        print("sort_pool_out_1 shape")
        print(sort_pool_out_1.shape)

        # hidden = self.conv2(graph, hidden)
        # print("conv2 out shape")
        # print(hidden.shape)
        sort_pool_out_1 = sort_pool_out_1.reshape(batch_size * self.sort_pool1.k, -1)
        hidden = self.conv2(batched_subgraph, sort_pool_out_1)
        print("conv2 out shape")
        print(hidden.shape)

        hidden = F.dropout(hidden, self.dropout, training=self.training)
        topk_node_idxs_batches, sort_pool_out_2 = self.sort_pool2(batched_subgraph, hidden)
        graphs = dgl.unbatch(batched_subgraph)
        subgraphs = []
        for g in range(batch_size):
            topk_node_idxs = topk_node_idxs_batches[g]
            subgraph = node_subgraph(graphs[g], topk_node_idxs)
            subgraphs.append(subgraph)
        batched_subgraph = dgl.batch(subgraphs)


       # x, edge_index, batch, unpool_info = edgePooling(hidden.to(self.device), edge_index.to(self.device), batch.to(self.device))
        #x, edge_index, batch, unpool_info = edgePooling(hidden, edge_index, batch)

        #hidden = self.conv3(graph, x)

        sort_pool_out_2 = sort_pool_out_2.reshape(batch_size * self.sort_pool2.k, -1)
        hidden = self.conv3(batched_subgraph, sort_pool_out_2)
        print("conv3 out shape")
        print(hidden.shape)
        hidden = F.dropout(hidden, self.dropout, training=self.training)
        out = self.conv4(batched_subgraph, hidden)
        print("conv4 out shape")
        print(out.shape)
        #hidden = F.dropout(hidden, self.dropout, training=self.training)
        return out, topk_node_idxs_batches

    def knn(self, x, y, k, batch_x=None, batch_y=None):
        if batch_x is None:
            batch_x = x.new_zeros(x.size(0), dtype=torch.long)

        if batch_y is None:
            batch_y = y.new_zeros(y.size(0), dtype=torch.long)

        x = x.view(-1, 1) if x.dim() == 1 else x
        y = y.view(-1, 1) if y.dim() == 1 else y

        assert x.dim() == 2 and batch_x.dim() == 1
        assert y.dim() == 2 and batch_y.dim() == 1
        assert x.size(1) == y.size(1)
        assert x.size(0) == batch_x.size(0)
        assert y.size(0) == batch_y.size(0)

        # Rescale x and y.
        min_xy = min(x.min().item(), y.min().item())
        x, y = x - min_xy, y - min_xy

        max_xy = max(x.max().item(), y.max().item())
        x, y, = x / max_xy, y / max_xy

        # Concat batch/features to ensure no cross-links between examples exist.
        x = torch.cat([x, 2 * x.size(1) * batch_x.view(-1, 1).to(x.dtype)], dim=-1)
        y = torch.cat([y, 2 * y.size(1) * batch_y.view(-1, 1).to(y.dtype)], dim=-1)

        # print("Before calling scipy.spatial.cKDTree()")
        tree = scipy.spatial.cKDTree(x.detach().cpu().numpy(), balanced_tree=False)
        # print("After calling scipy.spatial.cKDTree()")
        # print("Before calling tree.query")
        dist, col = tree.query(
            y.detach().cpu(), k=k, distance_upper_bound=x.size(1))
        # print("After calling tree.query")

        dist = torch.from_numpy(dist).to(x.dtype)
        col = torch.from_numpy(col).to(torch.long)
        row = torch.arange(col.size(0), dtype=torch.long).view(-1, 1).repeat(1, k)
        mask = ~torch.isinf(dist).view(-1)
        row, col = row.view(-1)[mask], col.view(-1)[mask]

        return torch.stack([row, col], dim=0)

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
