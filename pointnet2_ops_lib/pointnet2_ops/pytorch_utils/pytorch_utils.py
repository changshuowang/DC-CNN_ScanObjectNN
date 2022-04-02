import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from typing import List, Tuple
import math
import copy

class ScoreNet(nn.Module):
    def __init__(self, in_channel, out_channel, hidden_unit=[8], last_bn=False, temp=1, d_w = False):
        super(ScoreNet, self).__init__()
        self.hidden_unit = hidden_unit
        self.last_bn = last_bn
        self.mlp_convs_hidden = nn.ModuleList()
        self.mlp_bns_hidden = nn.ModuleList()
        self.temp = temp
        self.d_w = d_w

        hidden_unit = list() if hidden_unit is None else copy.deepcopy(hidden_unit)
        hidden_unit.append(out_channel)
        hidden_unit.insert(0, in_channel)

        for i in range(1, len(hidden_unit)):  # from 1st hidden to next hidden to last hidden
            self.mlp_convs_hidden.append(nn.Conv2d(hidden_unit[i - 1], hidden_unit[i], 1,
                                                   bias=False if i < len(hidden_unit) - 1 else not last_bn))
            self.mlp_bns_hidden.append(nn.BatchNorm2d(hidden_unit[i]))

    def forward(self, xyz, score_norm='softmax'):
        # xyz : B*3*N*K
        B, _, N, K = xyz.size()
        scores = xyz

        for i, conv in enumerate(self.mlp_convs_hidden):
            if i < len(self.mlp_convs_hidden) - 1:
                scores = F.relu(self.mlp_bns_hidden[i](conv(scores)))
            else:  # if the output layer, no ReLU
                scores = conv(scores)
                if self.last_bn:
                    scores = self.mlp_bns_hidden[i](scores)
        if score_norm == 'softmax':
            scores = F.softmax(scores/self.temp, dim=1)  # + 0.5  # B*m*N*K
        elif score_norm == 'sigmoid':
            scores = torch.sigmoid(scores/self.temp)  # + 0.5  # B*m*N*K
        elif score_norm is None:
            scores = scores
            
        else:
            raise ValueError('Not Implemented!')

        if self.d_w == False:
            scores = scores.permute(0, 2, 3, 1)  # B*N*K*m

        return scores

def assign_score(score, point_input):
    B, N, K, m = score.size()
    score = score.view(B, N, K, 1, m)
    point_output = torch.matmul(score, point_input).view(B, N, K, -1)  # b,n,k,cout
    return point_output

########## Dynamic-Cover Convolution begin ############
class DCConv(nn.Module):
    '''
    Input shape: (B, C_in, npoint, nsample)
    Output shape: (B, C_out, npoint)
    '''
    def __init__(
            self, 
            C_in, 
            C_out,
            activation = nn.ReLU(inplace=True),
            mapping = None,
            num_kernel = None,
            relation_prior = 1,
            first_layer = False
    ):
        super(DCConv, self).__init__()                                             
        self.bn_dcconv = nn.BatchNorm2d(C_in) if not first_layer else nn.BatchNorm2d(16)
        self.bn_channel_raising = nn.BatchNorm1d(C_out)
        self.bn_xyz_raising = nn.BatchNorm2d(16)
        if first_layer:
            self.bn_mapping = nn.BatchNorm2d(math.floor(C_out / 2))
        else: 
            self.bn_mapping = nn.BatchNorm2d(math.floor(C_out / 4))
        self.activation = activation
        self.relation_prior = relation_prior
        self.first_layer = first_layer
        self.mapping_func1 = mapping[0]
        self.mapping_func2 = mapping[1]
        self.cr_mapping = mapping[2]
        if first_layer:
            self.xyz_raising = mapping[3]

        self.p = torch.FloatTensor(1)
        torch.nn.init.uniform_(self.p, a=1, b=2)
        self.p = torch.nn.Parameter(self.p)
        
        self.num_kernel = num_kernel
        self.scorenet = ScoreNet(10, self.num_kernel, last_bn=False)

        if self.first_layer:
            hidden_unit = [8]
            self.h_xi_xj = ScoreNet(10, self.num_kernel * 16, hidden_unit, last_bn=False, d_w = True)
        else:
            hidden_unit = [math.floor(C_in / 2)]
            self.h_xi_xj = ScoreNet(10, self.num_kernel * C_in, hidden_unit, last_bn=False, d_w = True)
        
    def forward(self, input): # input: (B, 3 + 3 + C_in, npoint, centroid + nsample)
        
        x = input[:, 3:, :, :]           # (B, C_in, npoint, nsample+1), input features
        
        B, _, N, K = x.size()

        nsample = x.size()[3]
        if self.relation_prior == 2:
            abs_coord = input[:, 0:2, :, :]
            delta_x = input[:, 3:5, :, :]
            zero_vec = Variable(torch.zeros(x.size()[0], 1, x.size()[2], nsample).cuda())
        else:
            abs_coord = input[:, 0:3, :, :]  # (B, 3, npoint, nsample+1), absolute coordinates
            delta_x = input[:, 3:6, :, :]    # (B, 3, npoint, nsample+1), normalized coordinates
            
        coord_xi = abs_coord[:, :, :, 0:1].repeat(1, 1, 1, nsample)   # (B, 3, npoint, nsample),  centroid point

        h_xi_xj = torch.norm(delta_x, p = 2, dim = 1).unsqueeze(1)
       
        if self.relation_prior == 1:
            h_xi_xj = torch.cat((h_xi_xj, coord_xi, abs_coord, delta_x), dim = 1)
            # print("h_xi_xj", h_xi_xj, h_xi_xj.shape)
        elif self.relation_prior == 2:
            h_xi_xj = torch.cat((h_xi_xj, coord_xi, zero_vec, abs_coord, zero_vec, delta_x, zero_vec), dim = 1)
        del coord_xi, abs_coord, delta_x

        ##generation dynamic kernel
        scores = self.scorenet(h_xi_xj, score_norm='softmax')  # b,n,k,m
        self.d_w = self.h_xi_xj(h_xi_xj, score_norm= None)  # b, m * cin, n, k
        self.d_w = self.d_w.reshape(B, self.num_kernel, -1, N, K)  # B, m, cin, N1, K,
        del h_xi_xj

        if self.first_layer:
            x = self.activation(self.bn_xyz_raising(self.xyz_raising(x)))
        self.p.data = self.p.data.clamp(min=1, max=2)

        x = torch.mul(self.d_w, x.unsqueeze(1)).permute(0, 3, 4, 1, 2)  # b,n1,k,m,cin
        x = torch.mul(torch.pow(torch.abs(x + 0.000001) + 0.000001, self.p), torch.sign(x))
        
        x = assign_score(score=scores, point_input = x)  # b,n,k,cin,
        x = x.permute(0, 3, 1, 2)  # b,cin,n,k

        x = F.max_pool2d(self.activation(self.bn_dcconv(x.contiguous())), kernel_size=(1, nsample)).squeeze(3)  # (B, C_in, npoint)
        x = self.activation(self.bn_channel_raising(self.cr_mapping(x)))
        
        return x
        
class DCConvLayer(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            activation=nn.ReLU(inplace=True),
            conv=DCConv,
            mapping = None,
            num_kernel = None,
            relation_prior = 1,
            first_layer = False
    ):
        super(DCConvLayer, self).__init__()

        conv_unit = conv(
            in_size,
            out_size,
            activation = activation,
            mapping = mapping,
            num_kernel = num_kernel,
            relation_prior = relation_prior,
            first_layer = first_layer
        )

        self.add_module('DC_Conv', conv_unit)
                
class SharedDCConv(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            activation=nn.ReLU(inplace=True),
            mapping = None,
            num_kernel = None,
            relation_prior = 1,
            first_layer = False
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                'DCConvLayer{}'.format(i),
                DCConvLayer(
                    args[i],
                    args[i + 1],
                    activation = activation,
                    num_kernel = num_kernel,
                    mapping = mapping,
                    relation_prior = relation_prior,
                    first_layer = first_layer
                )
            )

########## Dynamic-Cover Convolution end ############



########## global convolutional pooling begin ############

class GloAvgConv(nn.Module):
    '''
    Input shape: (B, C_in, 1, nsample)
    Output shape: (B, C_out, npoint)
    '''
    def __init__(
            self, 
            C_in, 
            C_out, 
            init=nn.init.kaiming_normal_, 
            bias = True,
            activation = nn.ReLU(inplace=True)
    ):
        super(GloAvgConv, self).__init__()

        self.conv_avg = nn.Conv2d(in_channels = C_in, out_channels = C_out, kernel_size = (1, 1), 
                                  stride = (1, 1), bias = bias) 
        self.bn_avg = nn.BatchNorm2d(C_out)
        self.activation = activation
        
        init(self.conv_avg.weight)
        if bias:
            nn.init.constant_(self.conv_avg.bias, 0)
        
    def forward(self, x):
        nsample = x.size()[3]
        x = self.activation(self.bn_avg(self.conv_avg(x)))
        x = F.max_pool2d(x, kernel_size = (1, nsample)).squeeze(3)
        
        return x

########## global convolutional pooling end ############


class SharedMLP(nn.Sequential):

    def __init__(
            self,
            args: List[int],
            *,
            bn: bool = False,
            activation=nn.ReLU(inplace=True),
            preact: bool = False,
            first: bool = False,
            name: str = ""
    ):
        super().__init__()

        for i in range(len(args) - 1):
            self.add_module(
                name + 'layer{}'.format(i),
                Conv2d(
                    args[i],
                    args[i + 1],
                    bn=(not first or not preact or (i != 0)) and bn,
                    activation=activation
                    if (not first or not preact or (i != 0)) else None,
                    preact=preact
                )
            )
            

class _BNBase(nn.Sequential):

    def __init__(self, in_size, batch_norm=None, name=""):
        super().__init__()
        self.add_module(name + "bn", batch_norm(in_size))

        nn.init.constant_(self[0].weight, 1.0)
        nn.init.constant_(self[0].bias, 0)


class BatchNorm1d(_BNBase):

    def __init__(self, in_size: int, *, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm1d, name=name)


class BatchNorm2d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm2d, name=name)


class BatchNorm3d(_BNBase):

    def __init__(self, in_size: int, name: str = ""):
        super().__init__(in_size, batch_norm=nn.BatchNorm3d, name=name)


class _ConvBase(nn.Sequential):

    def __init__(
            self,
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=None,
            batch_norm=None,
            bias=True,
            preact=False,
            name=""
    ):
        super().__init__()

        bias = bias and (not bn)
        conv_unit = conv(
            in_size,
            out_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias
        )
        init(conv_unit.weight)
        if bias:
            nn.init.constant_(conv_unit.bias, 0)

        if bn:
            if not preact:
                bn_unit = batch_norm(out_size)
            else:
                bn_unit = batch_norm(in_size)

        if preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'conv', conv_unit)

        if not preact:
            if bn:
                self.add_module(name + 'bn', bn_unit)

            if activation is not None:
                self.add_module(name + 'activation', activation)


class Conv1d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: int = 1,
            stride: int = 1,
            padding: int = 0,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv1d,
            batch_norm=BatchNorm1d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv2d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int] = (1, 1),
            stride: Tuple[int, int] = (1, 1),
            padding: Tuple[int, int] = (0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv2d,
            batch_norm=BatchNorm2d,
            bias=bias,
            preact=preact,
            name=name
        )


class Conv3d(_ConvBase):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            kernel_size: Tuple[int, int, int] = (1, 1, 1),
            stride: Tuple[int, int, int] = (1, 1, 1),
            padding: Tuple[int, int, int] = (0, 0, 0),
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=nn.init.kaiming_normal_,
            bias: bool = True,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__(
            in_size,
            out_size,
            kernel_size,
            stride,
            padding,
            activation,
            bn,
            init,
            conv=nn.Conv3d,
            batch_norm=BatchNorm3d,
            bias=bias,
            preact=preact,
            name=name
        )


class FC(nn.Sequential):

    def __init__(
            self,
            in_size: int,
            out_size: int,
            *,
            activation=nn.ReLU(inplace=True),
            bn: bool = False,
            init=None,
            preact: bool = False,
            name: str = ""
    ):
        super().__init__()

        fc = nn.Linear(in_size, out_size, bias=not bn)
        if init is not None:
            init(fc.weight)
        if not bn:
            nn.init.constant_(fc.bias, 0)

        if preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(in_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

        self.add_module(name + 'fc', fc)

        if not preact:
            if bn:
                self.add_module(name + 'bn', BatchNorm1d(out_size))

            if activation is not None:
                self.add_module(name + 'activation', activation)

def set_bn_momentum_default(bn_momentum):

    def fn(m):
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = bn_momentum

    return fn

class BNMomentumScheduler(object):

    def __init__(
            self, model, bn_lambda, last_epoch=-1,
            setter=set_bn_momentum_default
    ):
        if not isinstance(model, nn.Module):
            raise RuntimeError(
                "Class '{}' is not a PyTorch nn Module".format(
                    type(model).__name__
                )
            )

        self.model = model
        self.setter = setter
        self.lmbd = bn_lambda

        self.step(last_epoch + 1)
        self.last_epoch = last_epoch

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1

        self.last_epoch = epoch
        self.model.apply(self.setter(self.lmbd(epoch)))

    def get_momentum(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
        return self.lmbd(epoch)