from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet2_ops_lib.pointnet2_ops import pointnet2_utils
import pointnet2_ops_lib.pointnet2_ops.pytorch_utils as pt_utils
import math


def build_shared_mlp(mlp_spec: List[int], bn: bool = True):
    layers = []
    for i in range(1, len(mlp_spec)):
        layers.append(
            nn.Conv2d(mlp_spec[i - 1], mlp_spec[i], kernel_size=1, bias=not bn)
        )
        if bn:
            layers.append(nn.BatchNorm2d(mlp_spec[i]))
        layers.append(nn.ReLU(True))

    return nn.Sequential(*layers)


class _PointnetSAModuleBase(nn.Module):
    def __init__(self):
        super(_PointnetSAModuleBase, self).__init__()
        self.npoint = None
        self.groupers = None
        self.mlps = None

    def forward(
        self, xyz: torch.Tensor, features: Optional[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        r"""
        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor of the xyz coordinates of the features
        features : torch.Tensor
            (B, C, N) tensor of the descriptors of the the features

        Returns
        -------
        new_xyz : torch.Tensor
            (B, npoint, 3) tensor of the new features' xyz
        new_features : torch.Tensor
            (B,  \sum_k(mlps[k][-1]), npoint) tensor of the new_features descriptors
        """

        new_features_list = []

        xyz_flipped = xyz.transpose(1, 2).contiguous()
        if self.npoint is not None:
            fps_idx = pointnet2_utils.furthest_point_sample(xyz, self.npoint)  # (B, npoint)
            new_xyz = pointnet2_utils.gather_operation(xyz_flipped, fps_idx).transpose(1, 2).contiguous()
            fps_idx = fps_idx.data
        else: 
            new_xyz = None
            fps_idx = None

        for i in range(len(self.groupers)):
            new_features = self.groupers[i](xyz, new_xyz, features, fps_idx) if self.npoint is not None else self.groupers[i](xyz, new_xyz, features)  # (B, C, npoint, nsample)

            new_features = self.mlps[i](new_features)  # (B, mlp[-1], npoint, nsample)
            # new_features = F.max_pool2d(new_features, kernel_size=[1, new_features.size(3)])  # (B, mlp[-1], npoint, 1)
            # new_features = new_features.squeeze(-1)  # (B, mlp[-1], npoint)

            new_features_list.append(new_features)

        return new_xyz, torch.cat(new_features_list, dim=1)


class PointnetSAModuleMSG(_PointnetSAModuleBase):
    r"""Pointnet set abstrction layer with multiscale grouping

    Parameters
    ----------
    npoint : int
        Number of features
    radii : list of float32
        list of radii to group with
    nsamples : list of int32
        Number of samples in each ball query
    mlps : list of list of int32
        Spec of the pointnet before the global max_pool for each scale
    bn : bool
        Use batchnorm
    """

    def __init__(self, npoint, radii, nsamples, mlps, num_kernel=4,bn=True, use_xyz=True, bias = True, init = nn.init.kaiming_normal_, first_layer = False, relation_prior = 1):
        # type: (PointnetSAModuleMSG, int, List[float], List[int], List[List[int]], bool, bool) -> None
        super(PointnetSAModuleMSG, self).__init__()

        assert len(radii) == len(nsamples) == len(mlps)

        self.npoint = npoint
        self.groupers = nn.ModuleList()
        self.mlps = nn.ModuleList()

        # initialize shared mapping functions
        C_in = (mlps[0][0] + 3) if use_xyz else mlps[0][0]
        C_out = mlps[0][1]
        
        if relation_prior == 0:
            in_channels = 1
        elif relation_prior == 1 or relation_prior == 2:
            in_channels = 10
        else:
            assert False, "relation_prior can only be 0, 1, 2."
        
        if first_layer:
            mapping_func1 = nn.Conv2d(in_channels = in_channels, out_channels = math.floor(C_out / 2), kernel_size = (1, 1), 
                                      stride = (1, 1), bias = bias)
            mapping_func2 = nn.Conv2d(in_channels = math.floor(C_out / 2), out_channels = 16, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
            xyz_raising = nn.Conv2d(in_channels = C_in, out_channels = 16, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
            init(xyz_raising.weight)
            if bias:
                nn.init.constant_(xyz_raising.bias, 0)
        elif npoint is not None:
            mapping_func1 = nn.Conv2d(in_channels = in_channels, out_channels = math.floor(C_out / 4), kernel_size = (1, 1), 
                                      stride = (1, 1), bias = bias)
            mapping_func2 = nn.Conv2d(in_channels = math.floor(C_out / 4), out_channels = C_in, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias)
        if npoint is not None:
            init(mapping_func1.weight)
            init(mapping_func2.weight)
            if bias:
                nn.init.constant_(mapping_func1.bias, 0)
                nn.init.constant_(mapping_func2.bias, 0)    
                     
            # channel raising mapping
            cr_mapping = nn.Conv1d(in_channels = C_in if not first_layer else 16, out_channels = C_out, kernel_size = 1, 
                                      stride = 1, bias = bias)
            init(cr_mapping.weight)
            nn.init.constant_(cr_mapping.bias, 0)
        
        if first_layer:
            mapping = [mapping_func1, mapping_func2, cr_mapping, xyz_raising]
        elif npoint is not None:
            mapping = [mapping_func1, mapping_func2, cr_mapping]
        
        for i in range(len(radii)):
            radius = radii[i]
            nsample = nsamples[i]
            self.groupers.append(
                pointnet2_utils.QueryAndGroup(radius, nsample, use_xyz=use_xyz)
                if npoint is not None else pointnet2_utils.GroupAll(use_xyz)
            )
            mlp_spec = mlps[i]
            if use_xyz:
                mlp_spec[0] += 3
            if npoint is not None:
                self.mlps.append(pt_utils.SharedDCConv(mlp_spec, mapping = mapping, num_kernel=num_kernel, relation_prior = relation_prior, first_layer = first_layer))
            else:   # global convolutional pooling
                self.mlps.append(pt_utils.GloAvgConv(C_in = C_in, C_out = C_out))

class PointnetSAModule(PointnetSAModuleMSG):
    r"""Pointnet set abstrction layer

    Parameters
    ----------
    npoint : int
        Number of features
    radius : float
        Radius of ball
    nsample : int
        Number of samples in the ball query
    mlp : list
        Spec of the pointnet before the global max_pool
    bn : bool
        Use batchnorm
    """

    def __init__(
        self, mlp, npoint=None, radius=None, nsample=None, bn=True, use_xyz=True
    ):
        # type: (PointnetSAModule, List[int], int, float, int, bool, bool) -> None
        super(PointnetSAModule, self).__init__(
            mlps=[mlp],
            npoint=npoint,
            radii=[radius],
            nsamples=[nsample],
            bn=bn,
            use_xyz=use_xyz,
        )


class PointnetFPModule(nn.Module):
    r"""Propigates the features of one set to another

    Parameters
    ----------
    mlp : list
        Pointnet module parameters
    bn : bool
        Use batchnorm
    """

    def __init__(self, mlp, bn=True):
        # type: (PointnetFPModule, List[int], bool) -> None
        super(PointnetFPModule, self).__init__()
        self.mlp = build_shared_mlp(mlp, bn=bn)

    def forward(self, unknown, known, unknow_feats, known_feats):
        # type: (PointnetFPModule, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor) -> torch.Tensor
        r"""
        Parameters
        ----------
        unknown : torch.Tensor
            (B, n, 3) tensor of the xyz positions of the unknown features
        known : torch.Tensor
            (B, m, 3) tensor of the xyz positions of the known features
        unknow_feats : torch.Tensor
            (B, C1, n) tensor of the features to be propigated to
        known_feats : torch.Tensor
            (B, C2, m) tensor of features to be propigated

        Returns
        -------
        new_features : torch.Tensor
            (B, mlp[-1], n) tensor of the features of the unknown features
        """

        if known is not None:
            dist, idx = pointnet2_utils.three_nn(unknown, known)
            dist_recip = 1.0 / (dist + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm

            interpolated_feats = pointnet2_utils.three_interpolate(
                known_feats, idx, weight
            )
        else:
            interpolated_feats = known_feats.expand(
                *(known_feats.size()[0:2] + [unknown.size(1)])
            )

        if unknow_feats is not None:
            new_features = torch.cat(
                [interpolated_feats, unknow_feats], dim=1
            )  # (B, C2 + C1, n)
        else:
            new_features = interpolated_feats

        new_features = new_features.unsqueeze(-1)
        new_features = self.mlp(new_features)

        return new_features.squeeze(-1)
