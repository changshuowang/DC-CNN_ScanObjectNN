import os, sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "../utils"))
import torch
import torch.nn as nn
from torch.autograd import Variable
import pointnet2_ops_lib.pointnet2_ops.pytorch_utils as pt_utils
from pointnet2_ops_lib.pointnet2_ops.pointnet2_modules import PointnetSAModule, PointnetSAModuleMSG
import numpy as np

# Dynamic-Cover CNN:  Multi-Scale Neighborhood
class DCCNN_MSN(nn.Module):
    r"""
        PointNet2 with multi-scale grouping
        Semantic segmentation network that uses feature propogation layers

        Parameters
        ----------
        num_classes: int
            Number of semantics classes to predict over -- size of softmax classifier that run for each point
        input_channels: int = 6
            Number of input channels in the feature descriptor for each point.  If the point cloud is Nx9, this
            value should be 6 as in an Nx9 point cloud, 3 of the channels are xyz, and 6 are feature descriptors
        use_xyz: bool = True
            Whether or not to use the xyz position of a point as a feature
    """

    def __init__(self, num_classes, num_kernel, input_channels=0, relation_prior=1, use_xyz=True):
        super().__init__()

        self.SA_modules = nn.ModuleList()
        
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=512,
                radii=[0.1, 0.2, 0.4],
                nsamples=[16, 32, 128],
                mlps=[[input_channels, 128], [input_channels, 128], [input_channels, 128]],
                num_kernel=num_kernel,
                first_layer=True,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_0 = 128 * 3

        c_in = c_out_0
        self.SA_modules.append(
            PointnetSAModuleMSG(
                npoint=128,
                radii=[0.2, 0.4, 0.8],
                nsamples=[32, 64, 128],
                mlps=[[c_in, 256], [c_in, 256], [c_in, 256]],
                num_kernel=num_kernel,
                use_xyz=use_xyz,
                relation_prior=relation_prior
            )
        )
        c_out_1 = 256 * 3

        self.SA_modules.append(
            # global convolutional pooling
            PointnetSAModule(
                nsample = 128,
                mlp=[c_out_1, 1024],
                use_xyz=use_xyz
            )
        )

        self.FC_layer = nn.Sequential(
            pt_utils.FC(1024, 512, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(512, 256, activation=nn.ReLU(inplace=True), bn=True),
            nn.Dropout(p=0.5),
            pt_utils.FC(256, num_classes, activation=None)
        )

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = (
            pc[..., 3:].transpose(1, 2).contiguous()
            if pc.size(-1) > 3 else None
        )
        return xyz, features

    def forward(self, pointcloud: torch.cuda.FloatTensor):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        xyz, features = self._break_up_pc(pointcloud)
        for module in self.SA_modules:
            xyz, features = module(xyz, features)
        return self.FC_layer(features.squeeze(-1))