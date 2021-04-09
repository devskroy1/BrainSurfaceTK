from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius, global_max_pool, knn
from models.pointnet.src.models.pytorch_utils import conv1d, conv2d, gather_nd

#from torch_points_kernels import knn

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ROOT_DIR = os.path.dirname(BASE_DIR)
# sys.path.append(os.path.join(ROOT_DIR, 'utils'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/sampling'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/grouping'))
# sys.path.append(os.path.join(ROOT_DIR, 'tf_ops/3d_interpolation'))
# from tf_interpolate import three_nn, three_interpolate
# import tf_grouping
# import tf_sampling
# import tf_util
# import nearest_neighbors.lib.python.nearest_neighbors as nearest_neighbors


def knn_query(k, support_pts, query_pts):
    """
    :param support_pts: points you have, B*N1*3
    :param query_pts: points you want to know the neighbour index, B*N2*3
    :param k: Number of neighbours in knn search
    :return: neighbor_idx: neighboring points indexes, B*N2*k
    """

    neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
    return neighbor_idx.astype(np.int32)

# def knn(x, k):
#     print("Inside knn function")
#     print("x shape")
#     print(x.shape)
#     inner = -2 * torch.matmul(x.transpose(1, 0), x)
#     xx = torch.sum(x ** 2, dim=1, keepdim=True)
#     print("inner shape")
#     print(inner.shape)
#     print("xx.shape")
#     print(xx.shape)
#     # pairwise_distance = -xx - inner - xx.transpose(2, 1)
#     pairwise_distance = -xx - inner - xx.transpose(1, 0)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx

def sampling(npoint, pts, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: npoint * D, sub-sampled point cloud
    '''
    # print("Inside sampling()")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = pts.size(0)
    batch_pts = torch.arange(0, batch_size).to(device)
    N = pts.size(1)
    # print("N")
    # print(N)
    # print("npoint")
    # print(npoint)
    # print("pts device")
    # print(pts.device)
    # print("batch_pts device")
    # print(batch_pts.device)

    fps_batch_idx = torch.zeros((batch_size, npoint), dtype=torch.int64)
    for b in range(batch_size):
        pts_batch = pts[b, :, :]
        fps_index = fps(x=pts_batch, batch=None, ratio=npoint/N)
        # print("fps_index shape")
        # print(fps_index.shape)
        #Should be npoint = 512

        fps_batch_idx[b] = fps_index

    #fps_index = fps(pts, batch_pts)

    #fps_idx = tf_sampling.farthest_point_sample(npoint, pts)

    batch_indices = np.tile(torch.reshape(batch_pts, (-1, 1, 1)).cpu().numpy(), (1, npoint,1))
    # print("batch indices shape")
    # print(batch_indices.shape)
    # print("fps_index")
    # print(fps_index)
    # print("fps_index.shape")
    # print(fps_index.shape)

    #expanded_fps_index = fps_index.unsqueeze(dim=1).unsqueeze(dim=2).expand(-1, npoint, 2)
    # print("unsqueezed fps_index.shape")
    # print(expanded_fps_index.shape)
    idx = torch.cat([torch.from_numpy(batch_indices).to(device), fps_batch_idx.unsqueeze(2)], dim=2)
    # print("idx shape before set_shape")

    # print("idx shape")
    # print(idx.shape)
    #Should be (32, 512, 2)

    #idx = idx.reshape(batch_size, npoint, 2)
    if feature is None:
        #return tf.gather_nd(pts, idx)
        return gather_nd(pts, idx)
        #return pts[list(idx.T)]

    else:
        #return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx)
        # print("Just before returning from sampling function")
        # print("pts shape")
        # print(pts.shape)
        # print("idx shape")
        # print(idx.shape)
        # print("feature shape")
        # print(feature.shape)
        #return pts[list(idx)], feature[list(idx)]
        return gather_nd(pts, idx), gather_nd(feature, idx)
        #return pts[list(idx.T)], feature[list(idx.T)]

def grouping(feature, K, src_xyz, q_xyz, use_xyz=True, use_knn=True, radius=0.2):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #print("Inside grouping function")
    batch_size = src_xyz.size(0)
    # print("Inside grouping function")
    # print("src_xyz shape")
    # print(src_xyz.shape)
    # print("q_xyz shape")
    # print(q_xyz.shape)
    # print("batch_size")
    # print(batch_size)
    ndataset = src_xyz.size(1)
    npoint = q_xyz.size(1)
    # print("npoint")
    # print(npoint)
    if use_knn:
        #point_indices = tf.py_func(knn_query, [K, src_xyz, q_xyz], tf.int32)

        #TODO: Use pytorch geometric knn function since torch points kernels knn function not implemented on CUDA
        #Work out how to get rid of batch dimension
        # print("src_xyz shape")
        # print(src_xyz.shape)
        # print("q_xyz shape")
        # print(q_xyz.shape)

        # print("npoint")
        # print(npoint)

        knn_out_batch_idx = torch.zeros((batch_size, npoint, K), dtype=torch.int64)
        for b in range(batch_size):
            src_xyz_batch = src_xyz[b, :, :]
            q_xyz_batch = q_xyz[b, :, :]
            knn_output_idxs, _ = knn(src_xyz_batch, q_xyz_batch, K)
            # print("knn_output_idx shape")
            # print(knn_output_idx.shape)
            # print("knn_output_dist shape")
            # print(knn_output_dist.shape)

            knn_out_batch_idx[b] = knn_output_idxs.reshape(npoint, K)

        # reshaped_src_xyz = src_xyz.reshape(batch_size * ndataset, 3)
        # reshaped_q_xyz = q_xyz.reshape(batch_size * npoint, 3)
        # point_indices = knn(reshaped_src_xyz, reshaped_q_xyz, K)
        point_indices = knn_out_batch_idx.unsqueeze(dim=3)
        # print("point_indices shape")

        # print(point_indices.shape)
        # point_indices = point_indices.reshape(-1, npoint, K, batch_size)
        # batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
        # idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        # idx.set_shape([batch_size, npoint, K, 2])
        # grouped_xyz = tf.gather_nd(src_xyz, idx)

        batch_indices = np.tile(torch.arange(0, batch_size).view(-1, 1, 1, 1).cpu().numpy(), (1, npoint, K, 1))

        # print("batch_indices shape")
        # print(batch_indices.shape)
        idx = torch.cat([torch.from_numpy(batch_indices).to(device), point_indices], dim=3)
        # print("src_xyz shape")
        # print(src_xyz.shape)
        # print("idx shape")
        # print(idx.shape)
        #idx = idx.reshape([batch_size, npoint, K, 2])

        #grouped_xyz = tf.gather_nd(src_xyz, idx)
        grouped_xyz = gather_nd(src_xyz, idx)
        #grouped_xyz = src_xyz[list(idx.T)]

        # if feature is None:
        #
        #     params_size = list(src_xyz.size())
        #
        #     assert len(idx.size()) == 2
        #     assert len(params_size) >= idx.size(1)
        #     # Generate indices
        #     idx = idx.t().long()
        #     ndim = idx.size(0)
        #     index = torch.zeros_like(idx[0]).long()
        #     m = 1
        #
        #     for i in range(ndim)[::-1]:
        #         index += idx[i] * m
        #         m *= src_xyz.size(i)
        #
        #     src_xyz = src_xyz.reshape((-1, *tuple(torch.tensor(src_xyz.size()[ndim:]))))
        #     grouped_xyz = src_xyz[index]

    #TODO: Implement gather_nd in pytorch
    #grouped_feature = tf.gather_nd(feature, idx)
    grouped_feature = gather_nd(feature, idx)
    #grouped_feature = feature[list(idx.T)]

    if use_xyz:
        grouped_feature = torch.cat([grouped_xyz, grouped_feature], dim=-1)

    return grouped_xyz, grouped_feature, idx

def weight_net_hidden(xyz, hidden_units, is_training, bn_decay=None, weight_decay = None, activation_fn=nn.ReLU()):

    net = xyz
    for i, num_hidden_units in enumerate(hidden_units):
        net = conv2d(net, num_hidden_units, kernel_size=[1,1], padding=0, stride=[1,1], bn=True,
                     is_training=is_training, activation_fn=activation_fn, bn_decay=bn_decay, weight_decay=weight_decay)
    return net

def nonlinear_transform(data_in, mlp, is_training, bn_decay=None, weight_decay = None, activation_fn=nn.ReLU()):

    #with tf.variable_scope(scope) as sc:

    net = data_in
    l = len(mlp)
    if l > 1:
        for i, out_ch in enumerate(mlp[0:(l-1)]):
            net = conv2d(net, out_ch, kernel_size=[1, 1],
                                padding=0, stride=[1, 1],
                                bn=True, is_training=is_training, activation_fn=nn.ReLU(),
                                bn_decay=bn_decay, weight_decay=weight_decay)

            #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear%d'%(i))
    net = conv2d(net, mlp[-1], kernel_size=[1, 1],
                        padding=0, stride=[1, 1],
                        bn=False, is_training=is_training, bn_decay=bn_decay,
                        activation_fn=nn.Sigmoid(), weight_decay=weight_decay)

    return net

def SampleWeights(new_point, grouped_xyz, mlps, is_training, bn_decay, weight_decay, bn=True, scaled=True):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
        grouped_xyz: (batch_size, npoint, nsample, 3)
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, 1)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    [batch_size, npoint, nsample, channel] = list(new_point.size())
    bottleneck_channel = max(32,channel//2)
    # print("bottleneck_channel")
    # print(bottleneck_channel)
    # normalized_xyz = grouped_xyz - tf.tile(torch.unsqueeze(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
    normalized_xyz = grouped_xyz.cpu().numpy() - np.tile(torch.unsqueeze(grouped_xyz[:, :, 0, :], 2).cpu().numpy(), (1, 1, nsample, 1))
    new_point = torch.cat([torch.from_numpy(normalized_xyz).to(device), new_point], dim=-1) # (batch_size, npoint, nsample, channel+3)

    # transformed_feature = nn.conv2d(new_point, bottleneck_channel * 2, [1, 1],
    #                                      padding='VALID', stride=[1, 1],
    #                                      bn=bn, is_training=is_training,
    #                                      scope='conv_kv_ds', bn_decay=bn_decay, weight_decay=weight_decay,
    #                                      activation_fn=None)
    #
    # print("new_point shape")
    # print(new_point.shape)
    transformed_feature = conv2d(new_point, bottleneck_channel * 2, kernel_size=[1, 1],
                                 padding=0, stride=[1,1], bn=bn, is_training=is_training,
                                 bn_decay=bn_decay, weight_decay=weight_decay,
                                 activation_fn=None)

    # transformed_new_point = nn.conv2d(new_point, bottleneck_channel, [1, 1],
    #                                        padding='VALID', stride=[1, 1],
    #                                        bn=bn, is_training=is_training,
    #                                        scope='conv_query_ds', bn_decay=bn_decay, weight_decay=weight_decay,
    #                                        activation_fn=None)

    transformed_new_point = conv2d(new_point, bottleneck_channel, kernel_size=[1, 1],
                                   padding=0, stride=[1,1],
                                   bn=bn, is_training=is_training,
                                   bn_decay=bn_decay, weight_decay=weight_decay,
                                   activation_fn=None)

    #Original code
    transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
    #New code
    #transformed_feature1 = transformed_feature[:, :bottleneck_channel, :, :]

    # Original code
    feature = transformed_feature[:, :, :, bottleneck_channel:]
    # New code
    #feature = transformed_feature[:, bottleneck_channel:, :, :]

    # print("transformed_new_point shape")
    # print(transformed_new_point.shape)
    # print("transformed_feature.shape")
    # print(transformed_feature.shape)
    # print("transformed_feature1.shape")
    # print(transformed_feature1.shape)
    weights = torch.matmul(transformed_new_point, transformed_feature1.transpose(2, -1))  # (batch_size, npoint, nsample, nsample)
    if scaled:
        bottleneck_channel_float = torch.as_tensor(bottleneck_channel, dtype=torch.float)
        weights = weights / torch.sqrt(bottleneck_channel_float)
    softmax = nn.Softmax(dim=-1)
    weights = softmax(weights)
    channel = bottleneck_channel

    new_group_features = torch.matmul(weights, feature)
    new_group_features = new_group_features.reshape(batch_size, npoint, nsample, channel)
    for i, c in enumerate(mlps):
        activation = nn.ReLU() if i < len(mlps) - 1 else None
        new_group_features = conv2d(new_group_features, c, kernel_size=[1, 1],
                                           padding=0, stride=[1, 1],
                                           bn=bn, is_training=is_training,
                                           bn_decay=bn_decay, weight_decay=weight_decay,
                                           activation_fn=activation)

    softmax = nn.Softmax(dim=2)
    new_group_weights = softmax(new_group_features)
    # new_group_weights = nn.softmax(new_group_features, axis=2)  # (batch_size, npoint,nsample, mlp[-1)
    return new_group_weights

def AdaptiveSampling(group_xyz, group_feature, num_neighbor, is_training, bn_decay, weight_decay, bn):
    # with tf.variable_scope(scope) as sc:
    # print("Inside AdaptiveSampling()")
    [nsample, num_channel] = list(group_feature.size()[-2:])
    # print("num_channel")
    # print(num_channel)
    if num_neighbor == 0:
        new_xyz = group_xyz[:, :, 0, :]
        new_feature = group_feature[:, :, 0, :]
        return new_xyz, new_feature
    # print("group_xyz shape")
    # print(group_xyz.shape)
    shift_group_xyz = group_xyz[:, :, :num_neighbor, :]
    # print("group_feature shape")
    # print(group_feature.shape)
    shift_group_points = group_feature[:, :, :num_neighbor, :]
    sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [32, 1 + num_channel], is_training, bn_decay, weight_decay, bn)
    # print("sample weights shape")
    # print(sample_weight.shape)
    # print("shift_group_points shape")
    # print(shift_group_points.shape)
    # new_weight_xyz = tf.tile(torch.unsqueeze(sample_weight[:,:,:, 0],-1), [1, 1, 1, 3])
    new_weight_xyz = np.tile(torch.unsqueeze(sample_weight[:, :, :, 0], -1).detach().numpy(), (1, 1, 1, 3))
    # print("new_weight_xyz shape")
    # print(new_weight_xyz.shape)
    # print("shift_group_xyz shape")
    # print(shift_group_xyz.shape)
    new_weight_feature = sample_weight[:,:,:, 1:]
    # print("new_weight_feature shape")
    # print(new_weight_feature.shape)
    new_shift_group_xyz = np.tile(torch.unsqueeze(shift_group_xyz[:, :, :, 0], -1).detach().numpy(), (1, 1, 1, 3))
    #new_xyz = torch.sum(torch.multiply(shift_group_xyz.expand(shift_group_xyz.size(0), shift_group_xyz.size(1), shift_group_xyz.size(2), 3), torch.from_numpy(new_weight_xyz)), dim=2)

    #new_xyz = torch.sum(torch.multiply(torch.from_numpy(new_shift_group_xyz), torch.from_numpy(new_weight_xyz)), dim=2)

    new_xyz = torch.sum(torch.multiply(shift_group_xyz, torch.from_numpy(new_weight_xyz)), dim=2)
    new_feature = torch.sum(torch.multiply(shift_group_points, new_weight_feature), dim=2)
    # print("new_xyz")
    # print(new_xyz)
    # print("new_feature")
    # print(new_feature)
    return new_xyz, new_feature

def PointNonLocalCell(feature,new_point,mlp,is_training, bn_decay, weight_decay, bn=True, scaled=True, mode='dot'):
    """Input
        feature: (batch_size, ndataset, channel) TF tensor
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, channel)
    """
    #with tf.variable_scope(scope) as sc:
    bottleneck_channel = mlp[0]
    [batch_size, npoint, nsample, channel] = list(new_point.size())
    ndataset = feature.size(1)
    feature = torch.unsqueeze(feature, dim=2) #(batch_size, ndataset, 1, channel)
    # transformed_feature = tf_util.conv2d(feature, bottleneck_channel * 2, [1,1],
    #                                         padding='VALID', stride=[1,1],
    #                                         bn=bn, is_training=is_training,
    #                                         scope='conv_kv', bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None)

    transformed_feature = conv2d(feature, bottleneck_channel * 2, kernel_size=[1, 1],
                                 padding=0, stride=[1, 1], bn=bn, is_training=is_training,
                                 bn_decay=bn_decay, weight_decay=weight_decay, activation_fn=None)

    # transformed_new_point = nn.conv2d(new_point, bottleneck_channel, [1,1],
    #                                         padding='VALID', stride=[1,1],
    #                                         bn=bn, is_training=is_training,
    #                                         scope='conv_query', bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None) #(batch_size, npoint, nsample, bottleneck_channel)

    transformed_new_point = conv2d(new_point, bottleneck_channel, kernel_size=[1,1],
                                   padding=0, stride=[1,1], bn=bn, is_training=is_training,
                                   bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None)

    transformed_new_point = torch.reshape(transformed_new_point, [batch_size, npoint*nsample, bottleneck_channel])
    transformed_feature1 = torch.squeeze(transformed_feature[:,:,:,:bottleneck_channel], dim=2) #(batch_size, ndataset, bottleneck_channel)
    transformed_feature2 = torch.squeeze(transformed_feature[:,:,:,bottleneck_channel:], dim=2) #(batch_size, ndataset, bottleneck_channel)
    if mode == 'dot':
        attention_map = torch.matmul(transformed_new_point, transformed_feature1) #(batch_size, npoint*nsample, ndataset)
        if scaled:
            bottleneck_channel = bottleneck_channel.type(torch.FloatTensor)
            attention_map = attention_map / torch.sqrt(bottleneck_channel)

    elif mode == 'concat':
        tile_transformed_feature1 = np.tile(torch.unsqueeze(transformed_feature1, dim=1).cpu().numpy(), (1,npoint*nsample,1,1)) # (batch_size,npoint*nsample, ndataset, bottleneck_channel)
        tile_transformed_new_point = np.tile(torch.reshape(transformed_new_point, (batch_size, npoint*nsample, 1, bottleneck_channel)).cpu().numpy(), (1,1,ndataset,1)) # (batch_size,npoint*nsample, ndataset, bottleneck_channel)
        merged_feature = torch.cat([torch.from_numpy(tile_transformed_feature1), torch.from_numpy(tile_transformed_new_point)], dim=-1)

        # attention_map = tf_util.conv2d(merged_feature, 1, [1,1],
        #                                     padding='VALID', stride=[1,1],
        #                                     bn=bn, is_training=is_training,
        #                                     scope='conv_attention_map', bn_decay=bn_decay, weight_decay = weight_decay)

        attention_map = conv2d(merged_feature, 1, kernel_size=[1,1],
                                  padding=0, stride=[1,1], bn=bn, is_training=is_training,
                                  bn_decay=bn_decay, weight_decay = weight_decay)

        attention_map = torch.reshape(attention_map, (batch_size, npoint*nsample, ndataset))
    softmax = nn.Softmax(dim=-1)
    attention_map = softmax(attention_map)
    new_nonlocal_point = torch.matmul(attention_map, transformed_feature2) #(batch_size, npoint*nsample, bottleneck_channel)
    # new_nonlocal_point = tf_util.conv2d(tf.reshape(new_nonlocal_point,[batch_size,npoint, nsample, bottleneck_channel]), mlp[-1], [1,1],
    #                                         padding='VALID', stride=[1,1],
    #                                         bn=bn, is_training=is_training,
    #                                         scope='conv_back_project', bn_decay=bn_decay, weight_decay = weight_decay)

    new_nonlocal_point = conv2d(torch.reshape(new_nonlocal_point,[batch_size,npoint, nsample, bottleneck_channel]),
                                mlp[-1], kernel_size=[1, 1], padding=0, stride=[1,1],
                                bn=bn, is_training=is_training, bn_decay=bn_decay, weight_decay = weight_decay)

    new_nonlocal_point = torch.squeeze(new_nonlocal_point, dim=1)  # (batch_size, npoints, mlp2[-1])

    return new_nonlocal_point

def PointASNLSetAbstraction(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, weight_decay, bn=True, use_knn=True, radius=None, as_neighbor=8, NL=True):
    ''' Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            feature: (batch_size, ndataset, channel) TF tensor
            point: int32 -- #points sampled in Euclidean space by farthest point sampling
            nsample: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
    '''
    #with tf.variable_scope(scope) as sc:

    [batch_size, num_points, num_channel] = list(feature.size())
    '''Farthest Point Sampling'''
    if num_points == npoint:
        new_xyz = xyz
        new_feature = feature
    else:
        new_xyz, new_feature = sampling(npoint, xyz, feature)

    grouped_xyz, new_point, idx = grouping(feature, nsample, xyz, new_xyz,use_knn=use_knn,radius=radius)
    nl_channel = mlp[-1]

    '''Adaptive Sampling'''
    if num_points != npoint:
        new_xyz, new_feature = AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay, weight_decay, bn)
    grouped_xyz -= np.tile(torch.unsqueeze(new_xyz, 2).cpu().numpy(), (1, 1, nsample, 1))  # translation normalization
    new_point = torch.cat([grouped_xyz, new_point], dim=-1)

    '''Point NonLocal Cell'''
    if NL:
        new_nonlocal_point = PointNonLocalCell(feature, torch.unsqueeze(new_feature, dim=1),
                                               [max(32, num_channel//2), nl_channel],
                                               is_training, bn_decay, weight_decay, bn)

    '''Skip Connection'''
    skip_spatial = torch.max(new_point, dim=2)

    # skip_spatial = tf_util.conv1d(skip_spatial, mlp[-1], 1,padding='VALID', stride=1,
    #                              bn=bn, is_training=is_training, scope='skip',
    #                              bn_decay=bn_decay, weight_decay=weight_decay)

    skip_spatial = conv1d(skip_spatial,  mlp[-1], kernel_size=1, padding=0, stride=1,
                          bn=bn, is_training=is_training, bn_decay=bn_decay, weight_decay=weight_decay)

    '''Point Local Cell'''
    for i, num_out_channel in enumerate(mlp):
        if i != len(mlp) - 1:
            # new_point = tf_util.conv2d(new_point, num_out_channel, [1,1],
            #                             padding='VALID', stride=[1,1],
            #                             bn=bn, is_training=is_training,
            #                             scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

            new_point = conv2d(new_point, num_out_channel, kernel_size=[1, 1],
                               padding=0, stride=[1,1], bn=bn, is_training=is_training,
                               bn_decay=bn_decay, weight_decay=weight_decay)

    weight = weight_net_hidden(grouped_xyz, [32], is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
    new_point = torch.transpose(new_point, 2, 3)
    new_point = torch.matmul(new_point, weight)
    new_point = conv2d(new_point, mlp[-1], kernel_size=[1,new_point.size(2)],
                                    padding=0, stride=[1,1], bn=bn, is_training=is_training,
                                    bn_decay=bn_decay, weight_decay=weight_decay)

    new_point = torch.squeeze(new_point, 2)  # (batch_size, npoints, mlp2[-1])

    # new_point = tf.add(new_point,skip_spatial)

    new_point = new_point + skip_spatial

    if NL:
        #new_point = tf.add(new_point, new_nonlocal_point)
        new_point = new_point + new_nonlocal_point

    '''Feature Fushion'''
    new_point = conv1d(new_point, mlp[-1], kernel_size=1,
                              padding=0, stride=1, bn=bn, is_training=is_training,
                              bn_decay=bn_decay, weight_decay=weight_decay)

    return new_xyz, new_point