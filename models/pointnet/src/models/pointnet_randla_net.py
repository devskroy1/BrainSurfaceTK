import torch
import torch.nn as nn
from torch_points_kernels import knn
#import torch_points_kernels.points_cpu as tpcpu
#from torch_geometric.nn import knn

#TODO: Remove comment. From DGCNN repo. Don't use this. Use torch_points_kernels knn function as it has correct return type
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
#     #pairwise_distance = -xx - inner - xx.transpose(2, 1)
#     pairwise_distance = -xx - inner - xx.transpose(1, 0)
#     idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
#     return idx

#TODO: Use this function
# def knn(pos_support, pos, k):
#     """Dense knn serach
#     Arguments:
#         pos_support - [B,N,3] support points
#         pos - [B,M,3] centre of queries
#         k - number of neighboors, needs to be > N
#     Returns:
#         idx - [B,M,k]
#         dist2 - [B,M,k] squared distances
#     """
#     print("pos_support dim")
#     print(pos_support.dim())
#     print("pos dim")
#     print(pos.dim())
#     assert pos_support.dim() == 3 and pos.dim() == 3
#     if pos_support.is_cuda:
#         raise ValueError("CUDA version not implemented, use pytorch geometric")
#     return tpcpu.dense_knn(pos_support, pos, k)

class SharedMLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=1,
        stride=1,
        transpose=False,
        padding_mode='zeros',
        bn=False,
        activation_fn=None
    ):
        super(SharedMLP, self).__init__()

        conv_fn = nn.ConvTranspose2d if transpose else nn.Conv2d

        self.conv = conv_fn(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding_mode=padding_mode
        )
        self.batch_norm = nn.BatchNorm2d(out_channels, eps=1e-6, momentum=0.99) if bn else None
        self.activation_fn = activation_fn

    def forward(self, input):
        r"""
            Forward pass of the network
            Parameters
            ----------
            input: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, K)
        """
        x = self.conv(input)
        if self.batch_norm:
            x = self.batch_norm(x)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LocalSpatialEncoding(nn.Module):
    def __init__(self, d, num_neighbors, device):
        super(LocalSpatialEncoding, self).__init__()

        self.num_neighbors = num_neighbors
        self.mlp = SharedMLP(10, d, bn=True, activation_fn=nn.ReLU())

        self.device = device

    def forward(self, coords, features, knn_output):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d, N, 1)
                features of the point cloud
            neighbors: tuple
            Returns
            -------
            torch.Tensor, shape (B, 2*d, N, K)
        """
        # print("Inside LocSE forward()")
        # print("coords shape")
        # print(coords.shape)
        # print("features shape")
        # print(features.shape)
        # print("knn_output shape")
        # print(knn_output.shape)
        B = coords.size(0)
        N = coords.size(1)
        K = self.num_neighbors
        # finding neighboring points
        # print("Inside LocSE forward()")
        # print("knn_output")
        # print(knn_output)
        idx, dist = knn_output
        # print("idx")
        # print(idx)
        # print("dist")
        # print(dist)

        #Reshapes needed for torch geometric knn
        idx = idx.reshape(B, N, K)
        dist = dist.reshape(B, N, K)

        # print("idx")
        # print(idx)
        # print("idx shape")
        # print(idx.shape)
        # print("dist")
        # print(dist)
        # print("dist shape")
        # print(dist.shape)

        #B, N, K = idx.size()
        # idx(B, N, K), coords(B, N, 3)
        # neighbors[b, i, n, k] = coords[b, idx[b, n, k], i] = extended_coords[b, i, extended_idx[b, i, n, k], k]
        extended_idx = idx.unsqueeze(1).expand(B, 3, N, K)
        extended_coords = coords.transpose(-2,-1).unsqueeze(-1).expand(B, 3, N, K)
        # print("extended_idx shape")
        # print(extended_idx.shape)
        # print("extended_coords shape")
        # print(extended_coords.shape)
        neighbors = torch.gather(extended_coords, 2, extended_idx) # shape (B, 3, N, K)
        # if USE_CUDA:
        #     neighbors = neighbors.cuda()

        # relative point position encoding
        concat = torch.cat((
            extended_coords,
            neighbors,
            extended_coords - neighbors,
            dist.unsqueeze(-3)
        ), dim=-3).to(self.device)
        return torch.cat((
            self.mlp(concat),
            features.expand(B, -1, N, K)
        ), dim=-3)



class AttentivePooling(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(AttentivePooling, self).__init__()

        self.score_fn = nn.Sequential(
            nn.Linear(in_channels, in_channels, bias=False),
            nn.Softmax(dim=-2)
        )
        self.mlp = SharedMLP(in_channels, out_channels, bn=True, activation_fn=nn.ReLU())

    def forward(self, x):
        r"""
            Forward pass
            Parameters
            ----------
            x: torch.Tensor, shape (B, d_in, N, K)
            Returns
            -------
            torch.Tensor, shape (B, d_out, N, 1)
        """
        # computing attention scores
        scores = self.score_fn(x.permute(0,2,3,1)).permute(0,3,1,2)

        # sum over the neighbors
        features = torch.sum(scores * x, dim=-1, keepdim=True) # shape (B, d_in, N, 1)

        return self.mlp(features)



class LocalFeatureAggregation(nn.Module):
    def __init__(self, d_in, d_out, num_neighbors, device):
        super(LocalFeatureAggregation, self).__init__()

        self.num_neighbors = num_neighbors

        self.mlp1 = SharedMLP(d_in, d_out//2, activation_fn=nn.LeakyReLU(0.2))
        self.mlp2 = SharedMLP(d_out, 2*d_out)
        self.shortcut = SharedMLP(d_in, 2*d_out, bn=True)

        self.lse1 = LocalSpatialEncoding(d_out//2, num_neighbors, device)
        self.lse2 = LocalSpatialEncoding(d_out//2, num_neighbors, device)

        self.pool1 = AttentivePooling(d_out, d_out//2)
        self.pool2 = AttentivePooling(d_out, d_out)

        self.lrelu = nn.LeakyReLU()

    def forward(self, coords, features):
        r"""
            Forward pass
            Parameters
            ----------
            coords: torch.Tensor, shape (B, N, 3)
                coordinates of the point cloud
            features: torch.Tensor, shape (B, d_in, N, 1)
                features of the point cloud
            Returns
            -------
            torch.Tensor, shape (B, 2*d_out, N, 1)
        """
        # print("Inside lfa forward()")
        # print("coords shape")
        # print(coords.shape)

        # print("num neighbours")
        # print(self.num_neighbors)
        # print("coords.cpu()")
        # print(coords.cpu())

        batch_size = coords.size(0)
        num_points = coords.size(1)

        knn_coords = coords.reshape(batch_size * num_points, 3)
        #Torch geometric knn function - use for CUDA
        #knn_output = knn(knn_coords, knn_coords, self.num_neighbors)

        #torch_points_kernels knn function - use only for CPU
        knn_output = knn(coords.cpu().contiguous(), coords.cpu().contiguous(), self.num_neighbors)

        # print("knn output")
        # print(knn_output)

        #Use for CUDA
        # print("knn output shape")
        # print(knn_output.shape)

        #DGCNN knn function - don't use this
        # knn_output = knn(coords.cpu().contiguous(), self.num_neighbors)
        # print("features")
        # print(features)
        # print("features.shape")
        # print(features.shape)
        #
        # print("knn_output")
        # print(knn_output)
        x = self.mlp1(features)

        x = self.lse1(coords, x, knn_output)
        x = self.pool1(x)

        x = self.lse2(coords, x, knn_output)
        x = self.pool2(x)

        return self.lrelu(self.mlp2(x) + self.shortcut(features))



class RandLANet(nn.Module):
    def __init__(self, d_in, num_classes, num_neighbors=16, decimation=4, device=torch.device('cpu')):
        super(RandLANet, self).__init__()
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_neighbors = num_neighbors
        self.decimation = decimation

        self.fc_start = nn.Linear(d_in, 8).to(device)
        self.bn_start = nn.Sequential(
            nn.BatchNorm2d(8, eps=1e-6, momentum=0.99),
            nn.LeakyReLU(0.2)
        ).to(device)

        # encoding layers
        self.encoder = nn.ModuleList([
            LocalFeatureAggregation(8, 16, num_neighbors, device),
            LocalFeatureAggregation(32, 64, num_neighbors, device),
            LocalFeatureAggregation(128, 128, num_neighbors, device),
            LocalFeatureAggregation(256, 256, num_neighbors, device)
        ]).to(device)

        self.mlp = SharedMLP(512, 512, activation_fn=nn.ReLU()).to(device)

        # decoding layers
        decoder_kwargs = dict(
            transpose=True,
            bn=True,
            activation_fn=nn.ReLU()
        )
        self.decoder = nn.ModuleList([
            SharedMLP(1024, 256, **decoder_kwargs),
            SharedMLP(512, 128, **decoder_kwargs),
            SharedMLP(256, 32, **decoder_kwargs),
            SharedMLP(64, 8, **decoder_kwargs)
        ]).to(device)

        # final semantic prediction
        self.fc_end = nn.Sequential(
            SharedMLP(8, 64, bn=True, activation_fn=nn.ReLU()),
            SharedMLP(64, 32, bn=True, activation_fn=nn.ReLU()),
            nn.Dropout(),
            SharedMLP(32, num_classes)
        ).to(device)
        self.device = device
        self.num_classes = num_classes
        # self = self.to(device)

    def forward(self, input):
        r"""
            Forward pass
            Parameters
            ----------
            input: torch.Tensor, shape (B, N, d_in)
                input points
            Returns
            -------
            torch.Tensor, shape (B, num_classes, N)
                segmentation scores for each point
        """
        # print("Inside pointnet_randla_net forward")
        # print("input shape")
        # print(input.shape)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input = input.to(device)

        #N = input.size(1)
        N = input.size(1)
        B = input.size(0)
        # print("Batch size B")
        # print(B)
        d_in = input.size(2)
        # print("input")
        # print(input)
        # print("N")
        # print(N)
        # print("B")
        # print(B)
        # print("d_in")
        # print(d_in)

        d = self.decimation
        # print("decimation")
        # print(d)

        #coords = input.pos.clone().cpu()

        #For CUDA
        # coords = input[..., :3]
        # local_features = input[..., 3:]

        #For CPU
        coords = input[..., :3].clone().cpu()
        local_features = input[..., 3:].clone().cpu()

        # print("coords shape")
        # print(coords.shape)
        # print("local_features shape")
        # print(local_features.shape)
        #coords = input[..., :3].clone().cpu()

        #input_expanded = input.x.unsqueeze(0).expand(2, -1, -1)
        #print("input_expanded shape")
        #print(input_expanded.shape)
        #x = self.fc_start(input[:10, :10, :10]).transpose(-2, -1).unsqueeze(-1)

        #Original code
        #x = self.fc_start(input).transpose(-2,-1).unsqueeze(-1)

        #New code
        x = self.fc_start(local_features).transpose(-2, -1).unsqueeze(-1)

        # print("Got past fc_start")
        # print("x shape")
        # print(x.shape)
        x = self.bn_start(x) # shape (B, d, N, 1)
        # print("Got past bn_start")
        # print("x shape")
        # print(x.shape)
        decimation_ratio = 1

        # <<<<<<<<<< ENCODER
        x_stack = []
        permutation = torch.randperm(N)
        # print("permutn")
        # print(permutation)
        # print("permutn size")
        # print(permutation.size())
        #coords = coords[permutation, :]

        coords = coords[:, permutation]
        # print("coords shape")
        # print(coords.shape)
        x = x[:,:,permutation]
        # print("x shape")
        # print(x.shape)

        for lfa in self.encoder:
            # at iteration i, x.shape = (B, N//(d**i), d_in)
            x = lfa(coords[:,:N//decimation_ratio], x)
            x_stack.append(x.clone())
            decimation_ratio *= d
            x = x[:,:,:N//decimation_ratio]


        # # >>>>>>>>>> ENCODER

        x = self.mlp(x)

        # <<<<<<<<<< DECODER
        for mlp in self.decoder:
            neighbors, _ = knn(
                coords[:,:N//decimation_ratio].cpu().contiguous(), # original set
                coords[:,:d*N//decimation_ratio].cpu().contiguous(), # upsampled set
                # coords[:, :N // decimation_ratio],  # original set
                # coords[:, :d * N // decimation_ratio],  # upsampled set
                1
            ) # shape (B, N, 1)
            neighbors = neighbors.to(self.device)

            extended_neighbors = neighbors.unsqueeze(1).expand(-1, x.size(1), -1, 1)

            x_neighbors = torch.gather(x, -2, extended_neighbors)

            x = torch.cat((x_neighbors, x_stack.pop()), dim=1)

            x = mlp(x)

            decimation_ratio //= d

        # >>>>>>>>>> DECODER
        # inverse permutation
        x = x[:,:,torch.argsort(permutation)]

        scores = self.fc_end(x)

        scores = scores.squeeze(-1)
        return scores.reshape(B*N, self.num_classes)
