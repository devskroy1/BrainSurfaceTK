import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.nn import PointConv, fps, radius, global_max_pool
from models.pointnet.src.models.pointasnl_util import PointASNLSetAbstraction, sampling, grouping, AdaptiveSampling, weight_net_hidden
from models.pointnet.src.models.pytorch_utils import conv1d, conv2d

class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        print("x shape")
        print(x.shape)
        print("pos shape")
        print(pos.shape)
        print("batch shape")
        print(batch.shape)

        idx = fps(pos, batch, ratio=self.ratio)
        print("idx shape")
        print(idx.shape)
        print("pos[idx] shape")
        print(pos[idx].shape)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)  # TODO: FIGURE OUT THIS WITH RESPECT TO NUMBER OF POINTS
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class Net(torch.nn.Module):
    def __init__(self, num_local_features, num_global_features):
        super(Net, self).__init__()

        self.num_global_features = num_global_features

        self.mlp = mlp
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 3+6 IS 3 FOR COORDINATES, 6 FOR FEATURES PER POINT.
        # self.sa1_module = SAModule(0.5, 0.2, MLP([3 + num_local_features, 64, 64, 96]))
        # self.sa1a_module = SAModule(0.5, 0.2, MLP([96 + 3, 96, 96, 128]))
        # self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        # self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))
        #
        # self.lin1 = Lin(1024 + num_global_features, 512)
        # self.lin2 = Lin(512, 256)
        # self.lin3 = Lin(256, 128)
        # self.lin4 = Lin(128, 1)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

        # New code for point asnl and pointnet++
        self.sa1_module = SAModule(0.5, 0.2, MLP([128 + num_local_features, 128, 128, 160]))
        self.sa1a_module = SAModule(0.5, 0.2, MLP([160 + 3, 160, 160, 192]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([192 + 3, 192, 192, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(1024 + num_global_features, 512)
        self.lin2 = Lin(512, 256)
        self.lin3 = Lin(256, 128)
        self.lin4 = Lin(128, 1)  # OUTPUT = NUMBER OF CLASSES, 1 IF REGRESSION TASK

    def forward(self, data):

        batch_size = data.y.size(0)

        num_points = data.x.size(0)
        # print("num_points")
        # print(num_points)
        num_local_features = data.x.size(1)
        num_coords = data.pos.size(1)

        quot = num_points // batch_size
        num_points_multiple = quot * batch_size

        data_x_slice = data.x[:num_points_multiple, :]
        data_pos_slice = data.pos[:num_points_multiple, :]

        features = data_x_slice.reshape(batch_size, quot, num_local_features)
        xyz = data_pos_slice.reshape(batch_size, quot, num_coords)

        # PointASNL SetAbstraction layer - Adaptive Sampling module

        new_xyz, new_feature = sampling(npoint=512, pts=xyz, feature=features)
        new_xyz = new_xyz.transpose(0, 1)
        new_feature = new_feature.transpose(0, 1)
        # print("After exiting from sampling")
        # print("new_xyz shape")
        # print(new_xyz.shape)
        # print("new_feature shape")
        # print(new_feature.shape)
        # print("Just before calling grouping()")
        # print("xyz shape")
        # print(xyz.shape)
        # print("new_xyz shape")
        # print(new_xyz.shape)

        grouped_xyz, new_point, idx = grouping(feature=features, K=32, src_xyz=xyz, q_xyz=new_xyz)
        grouped_xyz = grouped_xyz.transpose(0, 2)
        new_point = new_point.transpose(0, 2)
        # print("grouped_xyz shape after calling grouping()")
        # print(grouped_xyz.shape)
        # print("new_point shape after calling grouping()")
        # print(new_point.shape)
        # print("idx shape after calling grouping()")
        # print(idx.shape)

        # TODO: Need to update is_training depending on whether you're training model or evaluating model
        new_xyz, new_feature = AdaptiveSampling(group_xyz=grouped_xyz, group_feature=new_point, num_neighbor=12,
                                                is_training=True, bn=True, bn_decay=None, weight_decay=None)

        # print("After exiting from AdaptiveSampling")
        # print("new_xyz shape")
        # print(new_xyz.shape)
        # print("new_feature shape")
        # print(new_feature.shape)
        # grouped_xyz = np.tile(grouped_xyz.cpu().numpy(), (1, 1, 32, 1))

        # grouped_xyz -= np.tile(torch.unsqueeze(new_xyz, dim=2).detach().cpu().numpy(), (1, 1, 32, 1))  # translation normalization

        # expanded_new_xyz = torch.unsqueeze(new_xyz, dim=2)
        # grouped_xyz -= np.tile(expanded_new_xyz.cpu().detach().numpy(), (1, 1, 32, 1))  # translation normalization

        # grouped_xyz -= torch.repeat(expanded_new_xyz, (1, 1, 32, 1))  # translation normalization

        new_point = torch.cat([grouped_xyz, new_point], dim=-1)
        # print("new_point.shape after cat")
        # print(new_point.shape)
        '''Skip Connection'''
        skip_spatial_max, skip_spatial_idxs = torch.max(new_point, dim=2)
        # print("skip spatial max shape")
        # print(skip_spatial_max.shape)

        skip_spatial = conv1d(skip_spatial_max, self.mlp[-1], kernel_size=1, padding=0, stride=1,
                              bn=True, is_training=True)
        # print("skip_spatial shape")
        # print(skip_spatial.shape)

        weight = weight_net_hidden(grouped_xyz, [32], is_training=True)
        # print("weight shape from weight_net_hidden()")
        # print(weight.shape)
        new_point = new_point.transpose(2, 3)
        # print("new_point.shape after transpose")
        # print(new_point.shape)
        new_point = torch.matmul(new_point, weight)
        # print("new_point.shape after matmul with weight")
        # print(new_point.shape)
        new_point = conv2d(new_point, self.mlp[-1], kernel_size=[1, new_point.size(2)],
                           padding=0, stride=[1, 1], bn=True, is_training=True)
        # print("new_point.shape after conv2d")
        # print(new_point.shape)
        new_point = new_point.squeeze(2)  # (batch_size, npoints, mlp2[-1])
        # print("new_point.shape after squeeze")
        # print(new_point.shape)
        new_point = new_point + skip_spatial
        # print("new_point.shape after addition of skip_spatial")
        # print(new_point.shape)
        # print("new_point shape")
        # print(new_point.shape)
        # print("new_xyz shape")
        # print(new_xyz.shape)
        # print("data.batch")
        # print(data.batch)
        # POINTNET CLASSIFICN NETWORK

        # sa0_out = (data.x, data.pos, data.batch)
        first_dim = new_xyz.size(0)
        npoint = new_xyz.size(1)
        # print("batch size")
        # print(batch_size)
        # print("npoint")
        # print(npoint)

        # new_xyz shape: (Batch size, npoint, 3) = (32, 32, 3)

        new_xyz = new_xyz.to(self.device)
        new_point = new_point.to(self.device)

        print("batch_size")
        print(batch_size)
        print("npoint")
        print(npoint)

        print("new_xyz shape")
        print(new_xyz.shape)
        print("new_point shape")
        print(new_point.shape)

        new_point_third_dim = new_point.size(2)
        new_xyz_reshape = new_xyz.reshape(batch_size * npoint, 3)
        new_point_reshape = new_point.reshape(batch_size * npoint, new_point_third_dim)

        # new_xyz_batch0 = new_xyz[0, :, :]
        # new_xyz_batch1 = new_xyz[1, :, :]
        # new_xyz_concat = torch.cat([new_xyz_batch0, new_xyz_batch1], dim=0)
        # print("tensors equal")
        # print(torch.equal(new_xyz_reshape, new_xyz_concat))

        # print("npoint")
        # print(npoint)
        batch_cat_tensor = torch.zeros(npoint, dtype=torch.int64).to(self.device)
        for b in range(1, batch_size):
            batch_tensor = torch.ones(npoint, dtype=torch.int64).to(self.device) * b
            # batch_tensor = batch_tensor.new_full((, npoint), b)
            # print("batch_tensor shape")
            # print(batch_tensor.shape)
            # Expected to be 512
            batch_cat_tensor = torch.cat([batch_cat_tensor, batch_tensor], dim=0)

        # print("batch_cat_tensor shape")
        # print(batch_cat_tensor.shape)
        # Expected to be 1024
        # print("batch_cat_tensor")
        # print(batch_cat_tensor)

        # print("self.batch_size")
        # print(self.batch_size)
        # print("new_point_reshape shape")
        # print(new_point_reshape.shape)
        # print("new_xyz_reshape shape")
        # print(new_xyz_reshape.shape)

        sa0_out = (new_point_reshape, new_xyz_reshape, batch_cat_tensor)

        #sa0_out = (data.x, data.pos, data.batch)

        sa1_out = self.sa1_module(*sa0_out)
        sa1a_out = self.sa1a_module(*sa1_out)
        sa2_out = self.sa2_module(*sa1a_out)
        sa3_out = self.sa3_module(*sa2_out)
        x, pos, batch = sa3_out

        # Concatenates global features to the inputs.
        if self.num_global_features > 0:
            x = torch.cat((x, data.y[:, 1:self.num_global_features+1].view(-1, self.num_global_features)), 1)

        x = F.relu(self.lin1(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        # x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)
        return x.view(-1)
