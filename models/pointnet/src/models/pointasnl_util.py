from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
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


#def knn_query(k, support_pts, query_pts):
    # """
    # :param support_pts: points you have, B*N1*3
    # :param query_pts: points you want to know the neighbour index, B*N2*3
    # :param k: Number of neighbours in knn search
    # :return: neighbor_idx: neighboring points indexes, B*N2*k
    # """

    # neighbor_idx = nearest_neighbors.knn_batch(support_pts, query_pts, k, omp=True)
    # return neighbor_idx.astype(np.int32)

def knn(x, k):
    print("Inside knn function")
    print("x shape")
    print(x.shape)
    inner = -2 * torch.matmul(x.transpose(1, 0), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    print("inner shape")
    print(inner.shape)
    print("xx.shape")
    print(xx.shape)
    # pairwise_distance = -xx - inner - xx.transpose(2, 1)
    pairwise_distance = -xx - inner - xx.transpose(1, 0)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def sampling(npoint, pts, ratio, feature=None):
    '''
    inputs:
    npoint: scalar, number of points to sample
    pointcloud: B * N * D, input point cloud
    output:
    sub_pts: B * npoint * D, sub-sampled point cloud
    '''
    batch_size = pts.shape(0)
    fps_idx = fps(npoint, pts, ratio)
    #batch_indices = tf.tile(torch.reshape(torch.range(0, batch_size), (-1, 1, 1)), (1, npoint,1))

    batch_indices = torch.from_numpy(np.tile(torch.arange(0, batch_size).view(-1, 1, 1), (1, npoint,1)))

    idx = torch.concat([batch_indices, fps_idx.expand(2)], dim=2)
    idx = idx.reshape([batch_size, npoint, 2])
    if feature is None:

        params_size = list(pts.size())

        assert len(idx.size()) == 2
        assert len(params_size) >= idx.size(1)
        # Generate indices
        idx = idx.t().long()
        ndim = idx.size(0)
        index = torch.zeros_like(idx[0]).long()
        m = 1

        for i in range(ndim)[::-1]:
            index += idx[i] * m
            m *= pts.size(i)

        pts = pts.reshape((-1, *tuple(torch.tensor(pts.size()[ndim:]))))
        return pts[index]

    #     return tf.gather_nd(pts, idx)
    # else:
    #     return tf.gather_nd(pts, idx), tf.gather_nd(feature, idx)

def grouping(feature, K, src_xyz, q_xyz, use_xyz=True, use_knn=True, radius=0.2):
    '''
    K: neighbor size
    src_xyz: original point xyz (batch_size, ndataset, 3)
    q_xyz: query point xyz (batch_size, npoint, 3)
    '''

    batch_size = src_xyz.shape(0)
    npoint = q_xyz.get_shape(1)

    if use_knn:
        #point_indices = tf.py_func(knn_query, [K, src_xyz, q_xyz], tf.int32)

        point_indices = knn(src_xyz, K)

        # batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, npoint, K, 1))
        # idx = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
        # idx.set_shape([batch_size, npoint, K, 2])
        # grouped_xyz = tf.gather_nd(src_xyz, idx)

        batch_indices = torch.from_numpy(np.tile(torch.arange(0, batch_size).view(-1, 1, 1, 1), (1, npoint, K, 1)))

        idx = torch.concat([batch_indices, point_indices.expand(3)], dim=3)
        idx = idx.reshape([batch_size, npoint, K, 2])

        if feature is None:

            params_size = list(src_xyz.size())

            assert len(idx.size()) == 2
            assert len(params_size) >= idx.size(1)
            # Generate indices
            idx = idx.t().long()
            ndim = idx.size(0)
            index = torch.zeros_like(idx[0]).long()
            m = 1

            for i in range(ndim)[::-1]:
                index += idx[i] * m
                m *= src_xyz.size(i)

            src_xyz = src_xyz.reshape((-1, *tuple(torch.tensor(src_xyz.size()[ndim:]))))
            grouped_xyz = src_xyz[index]
    else:
        #TODO: Not sure how to implement query_ball_point with Pytorch
        point_indices, _ = tf_grouping.query_ball_point(radius, K, src_xyz, q_xyz)
        grouped_xyz = tf_grouping.group_point(src_xyz, point_indices)

    grouped_feature = tf.gather_nd(feature, idx)

    if use_xyz:
        grouped_feature = tf.concat([grouped_xyz, grouped_feature], axis=-1)

    return grouped_xyz, grouped_feature, idx

def weight_net_hidden(xyz, hidden_units, scope, is_training, bn_decay=None, weight_decay = None, activation_fn=tf.nn.relu):

    net = xyz
    mod_list = nn.ModuleList()
    for i, num_hidden_units in enumerate(hidden_units):
        conv = nn.conv2d(net, num_hidden_units, kernel_size=(1,1), padding=1, stride=1)
        bn = nn.BatchNorm2d()
        activn = nn.ReLU()
        mod_list.append(conv)
        mod_list.append(bn)
        mod_list.append(activn)
    return mod_list

def nonlinear_transform(data_in, mlp, scope, is_training, bn_decay=None, weight_decay = None, activation_fn = tf.nn.relu):

    with tf.variable_scope(scope) as sc:

        net = data_in
        l = len(mlp)
        if l > 1:
            for i, out_ch in enumerate(mlp[0:(l-1)]):
                net = tf_util.conv2d(net, out_ch, [1, 1],
                                    padding = 'VALID', stride=[1, 1],
                                    bn = True, is_training = is_training, activation_fn=tf.nn.relu,
                                    scope = 'nonlinear%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

                #net = tf_util.dropout(net, keep_prob=0.5, is_training=is_training, scope='dp_nonlinear%d'%(i))
        net = tf_util.conv2d(net, mlp[-1], [1, 1],
                            padding = 'VALID', stride=[1, 1],
                            bn = False, is_training = is_training,
                            scope = 'nonlinear%d'%(l-1), bn_decay=bn_decay,
                            activation_fn=tf.nn.sigmoid, weight_decay = weight_decay)

    return net

def SampleWeights(new_point, grouped_xyz, mlps, is_training, bn_decay, weight_decay, scope, bn=True, scaled=True):
    """Input
        grouped_feature: (batch_size, npoint, nsample, channel) TF tensor
        grouped_xyz: (batch_size, npoint, nsample, 3)
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, 1)
    """
    batch_size, npoint, nsample, channel = new_point.get_shape()
    bottleneck_channel = max(32,channel//2)
    # normalized_xyz = grouped_xyz - tf.tile(torch.unsqueeze(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
    normalized_xyz = grouped_xyz - np.tile(torch.unsqueeze(grouped_xyz[:, :, 0, :], 2), [1, 1, nsample, 1])
    new_point = torch.concat([normalized_xyz, new_point], axis=-1) # (batch_size, npoint, nsample, channel+3)

    mod_list = nn.ModuleList()

    # transformed_feature = nn.conv2d(new_point, bottleneck_channel * 2, [1, 1],
    #                                      padding='VALID', stride=[1, 1],
    #                                      bn=bn, is_training=is_training,
    #                                      scope='conv_kv_ds', bn_decay=bn_decay, weight_decay=weight_decay,
    #                                      activation_fn=None)

    transformed_feature = nn.conv2d(new_point, bottleneck_channel * 2, kernel_size=(1, 1), padding=1, stride=1)
    bn = nn.BatchNorm2d()
    mod_list.append(transformed_feature)
    mod_list.append(bn)

    # transformed_new_point = nn.conv2d(new_point, bottleneck_channel, [1, 1],
    #                                        padding='VALID', stride=[1, 1],
    #                                        bn=bn, is_training=is_training,
    #                                        scope='conv_query_ds', bn_decay=bn_decay, weight_decay=weight_decay,
    #                                        activation_fn=None)

    transformed_new_point = nn.conv2d(new_point, bottleneck_channel, kernel_size=(1, 1), padding=1, stride=1)
    bn = nn.BatchNorm2d()

    transformed_feature1 = transformed_feature[:, :, :, :bottleneck_channel]
    feature = transformed_feature[:, :, :, bottleneck_channel:]

    weights = torch.matmul(transformed_new_point, transformed_feature1, transpose_b=True)  # (batch_size, npoint, nsample, nsample)
    if scaled:
        bottleneck_channel = bottleneck_channel.type(torch.FloatTensor)
        weights = weights / torch.sqrt(bottleneck_channel)
    softmax = nn.Softmax(dim=-1)
    weights = softmax(weights)
    channel = bottleneck_channel

    new_group_features = torch.matmul(weights, feature)
    new_group_features = torch.reshape(new_group_features, (batch_size, npoint, nsample, channel))
    for i, c in enumerate(mlps):
        activation = nn.relu() if i < len(mlps) - 1 else None
        new_group_features = nn.conv2d(net, num_hidden_units, kernel_size=(1, 1), padding=1, stride=1)
        bn = nn.BatchNorm2d()
        # new_group_features = nn.conv2d(new_group_features, c, [1, 1],
        #                                    padding='VALID', stride=[1, 1],
        #                                    bn=bn, is_training=is_training,
        #                                    scope='mlp2_%d' % (i), bn_decay=bn_decay, weight_decay=weight_decay,
        #                                    activation_fn=activation)


    softmax = nn.Softmax(dim=2)
    new_group_weights = softmax(new_group_features)
    # new_group_weights = nn.softmax(new_group_features, axis=2)  # (batch_size, npoint,nsample, mlp[-1)
    return new_group_weights

def AdaptiveSampling(group_xyz, group_feature, num_neighbor, is_training, bn_decay, weight_decay, scope, bn):
    # with tf.variable_scope(scope) as sc:
    nsample, num_channel = group_feature.get_shape()[-2:]
    if num_neighbor == 0:
        new_xyz = group_xyz[:, :, 0, :]
        new_feature = group_feature[:, :, 0, :]
        return new_xyz, new_feature
    shift_group_xyz = group_xyz[:, :, :num_neighbor, :]
    shift_group_points = group_feature[:, :, :num_neighbor, :]
    sample_weight = SampleWeights(shift_group_points, shift_group_xyz, [32, 1 + num_channel], is_training, bn_decay, weight_decay, scope, bn)
    # new_weight_xyz = tf.tile(torch.unsqueeze(sample_weight[:,:,:, 0],-1), [1, 1, 1, 3])
    new_weight_xyz = np.tile(torch.unsqueeze(sample_weight[:, :, :, 0], -1), [1, 1, 1, 3])
    new_weight_feature = sample_weight[:,:,:, 1:]
    new_xyz = torch.sum(torch.multiply(shift_group_xyz, new_weight_xyz))
    new_feature = torch.sum(torch.multiply(shift_group_points, new_weight_feature))

    return new_xyz, new_feature

def PointNonLocalCell(feature,new_point,mlp,is_training, bn_decay, weight_decay, scope, bn=True, scaled=True, mode='dot'):
    """Input
        feature: (batch_size, ndataset, channel) TF tensor
        new_point: (batch_size, npoint, nsample, channel)
        Output
        (batch_size, npoint, nsample, channel)
    """
    #with tf.variable_scope(scope) as sc:
    bottleneck_channel = mlp[0]
    batch_size, npoint, nsample, channel = new_point.get_shape()
    ndataset = feature.get_shape()[1]
    feature = torch.unsqueeze(feature, dim=2) #(batch_size, ndataset, 1, channel)
    # transformed_feature = tf_util.conv2d(feature, bottleneck_channel * 2, [1,1],
    #                                         padding='VALID', stride=[1,1],
    #                                         bn=bn, is_training=is_training,
    #                                         scope='conv_kv', bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None)

    transformed_feature = nn.conv2d(feature, bottleneck_channel * 2, kernel_size=(1, 1), padding=1, stride=1)
    bn = nn.BatchNorm2d()

    # transformed_new_point = nn.conv2d(new_point, bottleneck_channel, [1,1],
    #                                         padding='VALID', stride=[1,1],
    #                                         bn=bn, is_training=is_training,
    #                                         scope='conv_query', bn_decay=bn_decay, weight_decay = weight_decay, activation_fn=None) #(batch_size, npoint, nsample, bottleneck_channel)

    transformed_new_point = nn.conv2d(new_point, bottleneck_channel, kernel_size=(1, 1), padding=1, stride=1)
    bn = nn.BatchNorm2d()

    transformed_new_point = torch.reshape(transformed_new_point, [batch_size, npoint*nsample, bottleneck_channel])
    transformed_feature1 = torch.squeeze(transformed_feature[:,:,:,:bottleneck_channel], dim=2) #(batch_size, ndataset, bottleneck_channel)
    transformed_feature2 = torch.squeeze(transformed_feature[:,:,:,bottleneck_channel:], dim=2) #(batch_size, ndataset, bottleneck_channel)
    if mode == 'dot':
        attention_map = torch.matmul(transformed_new_point, transformed_feature1) #(batch_size, npoint*nsample, ndataset)
        if scaled:
            bottleneck_channel = bottleneck_channel.type(torch.FloatTensor)
            attention_map = attention_map / torch.sqrt(bottleneck_channel)

    elif mode == 'concat':
        tile_transformed_feature1 = np.tile(torch.unsqueeze(transformed_feature1, dim=1),(1,npoint*nsample,1,1)) # (batch_size,npoint*nsample, ndataset, bottleneck_channel)
        tile_transformed_new_point = np.tile(torch.reshape(transformed_new_point, (batch_size, npoint*nsample, 1, bottleneck_channel)), (1,1,ndataset,1)) # (batch_size,npoint*nsample, ndataset, bottleneck_channel)
        merged_feature = torch.concat([tile_transformed_feature1,tile_transformed_new_point], dim=-1)

        # attention_map = tf_util.conv2d(merged_feature, 1, [1,1],
        #                                     padding='VALID', stride=[1,1],
        #                                     bn=bn, is_training=is_training,
        #                                     scope='conv_attention_map', bn_decay=bn_decay, weight_decay = weight_decay)

        attention_map = nn.conv2d(new_point, bottleneck_channel, kernel_size=(1, 1), padding=1, stride=1)

        attention_map = torch.reshape(attention_map, (batch_size, npoint*nsample, ndataset))
    softmax = nn.Softmax(dim=-1)
    attention_map = softmax(attention_map)
    new_nonlocal_point = torch.matmul(attention_map, transformed_feature2) #(batch_size, npoint*nsample, bottleneck_channel)
    # new_nonlocal_point = tf_util.conv2d(tf.reshape(new_nonlocal_point,[batch_size,npoint, nsample, bottleneck_channel]), mlp[-1], [1,1],
    #                                         padding='VALID', stride=[1,1],
    #                                         bn=bn, is_training=is_training,
    #                                         scope='conv_back_project', bn_decay=bn_decay, weight_decay = weight_decay)

    new_nonlocal_point = nn.conv2d(torch.reshape(new_nonlocal_point,[batch_size,npoint, nsample, bottleneck_channel]), mlp[-1], kernel_size=(1, 1), padding=1, stride=1)

    new_nonlocal_point = torch.squeeze(new_nonlocal_point, dim=1)  # (batch_size, npoints, mlp2[-1])

    return new_nonlocal_point

def PointASNLSetAbstraction(xyz, feature, npoint, nsample, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_knn=True, radius=None, as_neighbor=8, NL=True):
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

    batch_size, num_points, num_channel = feature.get_shape()
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
        new_xyz, new_feature = AdaptiveSampling(grouped_xyz, new_point, as_neighbor, is_training, bn_decay, weight_decay, scope, bn)
    grouped_xyz -= np.tile(torch.unsqueeze(new_xyz, 2), [1, 1, nsample, 1])  # translation normalization
    new_point = torch.concat([grouped_xyz, new_point], dim=-1)

    '''Point NonLocal Cell'''
    if NL:
        new_nonlocal_point = PointNonLocalCell(feature, tf.expand_dims(new_feature, axis=1),
                                               [max(32, num_channel//2), nl_channel],
                                               is_training, bn_decay, weight_decay, scope, bn)

    '''Skip Connection'''
    skip_spatial = torch.max(new_point, dim=2)

    # skip_spatial = tf_util.conv1d(skip_spatial, mlp[-1], 1,padding='VALID', stride=1,
    #                              bn=bn, is_training=is_training, scope='skip',
    #                              bn_decay=bn_decay, weight_decay=weight_decay)

    skip_spatial = nn.conv1d(skip_spatial,  mlp[-1], kernel_size=(1, 1), padding=1, stride=1)

    '''Point Local Cell'''
    for i, num_out_channel in enumerate(mlp):
        if i != len(mlp) - 1:
            # new_point = tf_util.conv2d(new_point, num_out_channel, [1,1],
            #                             padding='VALID', stride=[1,1],
            #                             bn=bn, is_training=is_training,
            #                             scope='conv%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)

            new_point = nn.conv2d(new_point, num_out_channel, kernel_size=(1, 1), padding=1, stride=1)

    weight = weight_net_hidden(grouped_xyz, [32], scope = 'weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)
    new_point = torch.transpose(new_point, [0, 1, 3, 2])
    new_point = torch.matmul(new_point, weight)
    new_point = tf_util.conv2d(new_point, mlp[-1], [1,new_point.get_shape()[2].value],
                                    padding='VALID', stride=[1,1],
                                    bn=bn, is_training=is_training,
                                    scope='after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

    new_point = tf.squeeze(new_point, [2])  # (batch_size, npoints, mlp2[-1])

    new_point = tf.add(new_point,skip_spatial)

    if NL:
        new_point = tf.add(new_point, new_nonlocal_point)

    '''Feature Fushion'''
    new_point = tf_util.conv1d(new_point, mlp[-1], 1,
                              padding='VALID', stride=1, bn=bn, is_training=is_training,
                              scope='aggregation', bn_decay=bn_decay, weight_decay=weight_decay)

    return new_xyz, new_point

def PointASNLDecodingLayer(xyz1, xyz2, points1, points2, nsample, mlp, is_training, bn_decay, weight_decay, scope, bn=True, use_xyz = True,use_knn=True, radius=None, dilate_rate=1, mode='concat', NL=False):
    ''' Input:
            xyz1: (batch_size, ndataset1, 3) TF tensor
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            K: int32 -- how many points in each local region
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        batch_size, num_points, num_channel = points2.get_shape()
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0/dist),axis=2,keepdims=True)
        norm = tf.tile(norm,[1,1,3])
        weight = (1.0/dist) / norm

        '''Point NonLocal Cell'''
        if NL:
            new_nonlocal_point = PointNonLocalCell(points1, tf.expand_dims(points2, axis=1), [max(32,num_channel),num_channel],
                                                       is_training, bn_decay, weight_decay, scope, bn, mode=mode)
            new_nonlocal_point = tf.squeeze(new_nonlocal_point, [1])  # (batch_size, npoints, mlp2[-1])
            points2 = tf.add(points2, new_nonlocal_point)

        interpolated_points = three_interpolate(points2, idx, weight)

        '''Point Local Cell'''
        grouped_xyz, grouped_feature, idx = grouping(interpolated_points, nsample, xyz1, xyz1, use_xyz=use_xyz,use_knn=use_knn, radius=radius)
        grouped_xyz -= tf.tile(tf.expand_dims(xyz1, 2), [1, 1, nsample, 1])  # translation normalization

        weight = weight_net_hidden(grouped_xyz, [32], scope = 'decode_weight_net', is_training=is_training, bn_decay = bn_decay, weight_decay = weight_decay)

        new_points = grouped_feature
        new_points = tf.transpose(new_points, [0, 1, 3, 2])

        new_points = tf.matmul(new_points, weight)

        new_points = tf_util.conv2d(new_points, mlp[0], [1,new_points.get_shape()[2].value],
                                        padding='VALID', stride=[1,1],
                                        bn=bn, is_training=is_training,
                                        scope='decode_after_conv', bn_decay=bn_decay, weight_decay = weight_decay)

        if points1 is not None:
            new_points1 = tf.concat(axis=-1, values=[new_points, tf.expand_dims(points1, axis = 2)]) # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = new_points

        for i, num_out_channel in enumerate(mlp):
            if i != 0:
                new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1,1],
                                            padding='VALID', stride=[1,1],
                                            bn=bn, is_training=is_training,
                                            scope='conv_%d'%(i), bn_decay=bn_decay, weight_decay = weight_decay)
        new_points = tf.squeeze(new_points1, [2]) # B,ndataset1,mlp[-1]

        return new_points

def placeholder_inputs(batch_size, num_point, channel):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    feature_pts_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, channel))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size, num_point))
    return pointclouds_pl, feature_pts_pl, labels_pl



def get_repulsion_loss(pred, nsample=20, radius=0.07):
    # pred: (batch_size, npoint,3)
    idx, pts_cnt = tf_grouping.query_ball_point(radius, nsample, pred, pred)
    tf.summary.histogram('smooth/unque_index', pts_cnt)

    grouped_pred = tf_grouping.group_point(pred, idx)  # (batch_size, npoint, nsample, 3)
    grouped_pred -= tf.expand_dims(pred, 2)

    ##get the uniform loss
    h = 0.03
    dist_square = tf.reduce_sum(grouped_pred ** 2, axis=-1)
    dist_square, idx = tf.nn.top_k(-dist_square, 5)
    dist_square = -dist_square[:, :, 1:]  # remove the first one
    dist_square = tf.maximum(1e-12,dist_square)
    dist = tf.sqrt(dist_square)
    weight = tf.exp(-dist_square/h**2)
    uniform_loss = tf.reduce_mean(radius-dist*weight)
    return uniform_loss