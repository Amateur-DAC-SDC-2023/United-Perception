import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from up.utils.general.registry_factory import MODULE_ZOO_REGISTRY
from up.utils.model.initializer import init_bias_focal
from up.utils.model.normalize import build_norm_layer
from up.utils.model.act_fn import build_act_fn
# from up.tasks.det.plugins.yolov5.models.components import Focus

__all__ = ['SkrNetRetina']


class ConvBnAct(nn.Module):
    # CBH/CBL
    def __init__(self,
                 in_planes,
                 out_planes,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'}):
        super(ConvBnAct, self).__init__()
        if padding is None:
            padding = kernel_size // 2 if isinstance(kernel_size, int) else [x // 2 for x in kernel_size]
        self.norm1_name, norm1 = build_norm_layer(out_planes, normalize, 1)
        self.act1_name, act1 = build_act_fn(act_fn)

        self.conv = nn.Conv2d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              groups=groups,
                              bias=False)
        self.add_module(self.norm1_name, norm1)
        self.add_module(self.act1_name, act1)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def act1(self):
        return getattr(self, self.act1_name)

    def forward(self, x):
        x = self.conv(x)
        x = self.norm1(x)
        x = self.act1(x)
        return x

class Space2Depth(nn.Module):
    def __init__(self, block_size):
        super(Space2Depth, self).__init__()
        self.bs = block_size

    def forward(self, x):
        N, C, H, W = x.size()
        x = x.view(N, C, H // self.bs, self.bs, W // self.bs, self.bs)  # (N, C, H//bs, bs, W//bs, bs)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # (N, bs, bs, C, H//bs, W//bs)
        x = x.view(N, C * (self.bs ** 2), H // self.bs, W // self.bs)  # (N, C*bs^2, H//bs, W//bs)
        return x


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_size=2,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'},
                 num_down=1):
        super(Focus, self).__init__()
        self.space2depth = Space2Depth(block_size)
        ratio_c = (block_size ** 2) ** num_down
        self.num_down = num_down
        # self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = ConvBnAct(in_planes * ratio_c, out_planes, kernel_size, stride,
                                    padding, groups, normalize=normalize, act_fn=act_fn)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        # x = self.space2depth(x)
        for _ in range(self.num_down):
            x = self.space2depth(x)
        x = self.conv_block(x)
        return x

class FocusDouble(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_size=4,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'}):
        super(FocusDouble, self).__init__()
        self.space2depth = Space2Depth(block_size)
        self.conv_block = ConvBnAct(in_planes * (block_size ** 2), out_planes, kernel_size, stride,
                                    padding, groups, normalize=normalize, act_fn=act_fn)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.space2depth(x)
        x = self.conv_block(x)
        return x

class FocusMaxPool(nn.Module):
    # Focus wh information into c-space
    def __init__(self,
                 in_planes,
                 out_planes,
                 block_size=2,
                 kernel_size=1,
                 stride=1,
                 padding=None,
                 groups=1,
                 normalize={'type': 'solo_bn'},
                 act_fn={'type': 'Hardswish'}):
        super(FocusMaxPool, self).__init__()
        self.space2depth = Space2Depth(block_size)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_block = ConvBnAct(in_planes * (block_size ** 2), out_planes, kernel_size, stride,
                                    padding, groups, normalize=normalize, act_fn=act_fn)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        x = self.space2depth(x)
        x = self.maxpool(x)
        x = self.conv_block(x)
        return x


class Scale(nn.Module):
    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        return input * self.scale

class ReorgLayer(nn.Module):
    def __init__(self, stride=2):
        super(ReorgLayer, self).__init__()
        self.stride = stride
    def forward(self, x):
        stride = self.stride
        assert(x.data.dim() == 4)
        B, C, H, W = x.shape
        assert(H % stride == 0)
        assert(W % stride == 0)
        ws = stride
        hs = stride
        x = x.view([B, C, H//hs, hs, W//ws, ws]).transpose(3, 4).contiguous()
        x = x.view([B, C, H//hs*W//ws, hs*ws]).transpose(2, 3).contiguous()
        x = x.view([B, C, hs*ws, H//hs, W//ws]).transpose(1, 2).contiguous()
        x = x.view([B, hs*ws*C, H//hs, W//ws])
        return x


@MODULE_ZOO_REGISTRY.register('SkrNet_retina')
class SkrNetRetina(nn.Module):
    def __init__(self, cfg=[96,192,384,768,1024,1024], out_strides=[8], num_anchor=2, class_activation='sigmoid', init_prior=0.01, num_classes=1, detection = True, in_channel=3, gt_encode=True, noscale=True, bypass=True, focus_type='none', normalize={'type': 'solo_bn'}, act_fn={'type': 'ReLU'}, head_bias=True, with_backbone=False, out_planes=3):
        super(SkrNetRetina, self).__init__()
        self.cfg = cfg
        if not with_backbone:
            self.bbox = nn.Conv2d(cfg[5], 4*num_anchor, 1, 1, bias=head_bias)
            self.score = nn.Conv2d(cfg[5], 1*num_anchor, 1, 1, bias=head_bias)
            self.scales = nn.ModuleList([Scale(init_value=1.0)])
        self.noscale = noscale
        # self.scales = nn.ModuleList([Scale(init_value=1.0)])
        self.relu = nn.ReLU(inplace=True)
        self.gt_encode = gt_encode
        self.bypass = bypass
        self.head_bias = head_bias
        self.with_backbone = with_backbone
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        if out_strides[0] == 8:
            self.reorg = ReorgLayer(stride=2)
        elif out_strides[0] == 4:
            self.reorg = conv_dw(self.cfg[2], 4 * self.cfg[2], stride=1)

        if focus_type == 'maxpool':
            self.model_p1 = nn.Sequential(
                # conv_dw(in_channel, self.cfg[0], 1),    #dw1
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(in_channel, self.cfg[0], 1),
                conv_dw(self.cfg[0], self.cfg[1], 1),   #dw2
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[1], self.cfg[2], 1),   #dw3
            )
        elif focus_type == 'v5':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0],
                      kernel_size=1, normalize=normalize, act_fn=act_fn),
                conv_dw(self.cfg[0], self.cfg[1], 1),   #dw2
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[1], self.cfg[2], 1),   #dw3
            )
        elif focus_type == 'v5_double':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0], block_size=4,
                      kernel_size=1, normalize=normalize, act_fn=act_fn),
                # Focus(in_channel, self.cfg[0],
                #       kernel_size=1, normalize=normalize, act_fn=act_fn, num_down=2),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        elif focus_type == 'v5_three':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0], block_size=8,
                      kernel_size=1, normalize=normalize, act_fn=act_fn),
                # Focus(in_channel, self.cfg[0],
                #       kernel_size=1, normalize=normalize, act_fn=act_fn, num_down=2),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        elif focus_type == 'v5_v5':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0],
                      kernel_size=1, normalize=normalize, act_fn=act_fn, num_down=2),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        elif focus_type == 'v5_maxpool':
            self.model_p1 = nn.Sequential(
                FocusMaxPool(in_channel, self.cfg[0],
                             kernel_size=1, normalize=normalize, act_fn=act_fn),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        else:
            self.model_p1 = nn.Sequential(
                conv_dw(in_channel, self.cfg[0], 1),    #dw1
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[0], self.cfg[1], 1),   #dw2
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[1], self.cfg[2], 1),   #dw3
            )    

        if out_strides[0] == 8:
            self.model_p2 = nn.Sequential(    
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[2], self.cfg[3], 1),   #dw4
                conv_dw(self.cfg[3], self.cfg[4], 1),   #dw5
            )
        elif out_strides[0] == 4:
            self.model_p2 = nn.Sequential(
                # nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[2], self.cfg[3], 1),   #dw4
                conv_dw(self.cfg[3], self.cfg[4], 1),   #dw5
            )
        if self.bypass:
            self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
                conv_dw(self.cfg[2] * 4 + self.cfg[4], self.cfg[5], 1),
            )
        else:
            self.model_p3 = nn.Sequential(
                conv_dw(self.cfg[4], self.cfg[5], 1),
            )
        self.initialize_weights()
        self.out_strides = out_strides
        self.out_planes = out_planes
        if not with_backbone and head_bias:
            init_bias_focal(self.score, class_activation, init_prior, num_classes)

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)

    def get_outplanes(self):
        """
        get dimension of the output tensor
        """
        return self.out_planes

    def forward(self, input):
        x = input['image']
        x_p1 = self.model_p1(x)
        x_p1_reorg = self.reorg(x_p1)
        x_p2 = self.model_p2(x_p1)
        if self.bypass:
            x_p3_in = torch.cat([x_p1_reorg, x_p2], 1)
        else:
            x_p3_in = x_p2
        x = self.model_p3(x_p3_in)
        if not self.with_backbone:
            if self.gt_encode:
                bbox = self.bbox(x)
            else:
                if self.noscale:
                    # bbox = self.relu(self.bbox(x))
                    bbox = self.bbox(x)
                else:
                    # bbox = self.relu(self.scales[0](self.bbox(x)))
                    bbox = self.scales[0](self.bbox(x))
                # bbox = self.relu(self.bbox(x))
            score = self.score(x)
            return {'features': [[score, bbox]], 'strides': self.get_outstrides()}
        else:
            return {'features': [x], 'strides': self.get_outstrides()}

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  


@MODULE_ZOO_REGISTRY.register('SkrNet_retina_single')
class SkrNetRetinaSingle(nn.Module):
    def __init__(self, cfg=[96,192,384,768,1024,1024], out_strides=[16], num_anchor=2, class_activation='sigmoid', init_prior=0.01, num_classes=1, detection = True, in_channel=3, gt_encode=True, noscale=False, bypass=True, focus_type='none', normalize={'type': 'solo_bn'}, act_fn={'type': 'ReLU'}, head_bias=True):
        super(SkrNetRetinaSingle, self).__init__()
        self.cfg = cfg
        self.reorg = ReorgLayer(stride=2)
        # self.bbox = nn.Conv2d(cfg[5], 4*num_anchor, 1, 1)
        # self.score = nn.Conv2d(cfg[5], 1*num_anchor, 1, 1)
        self.head = nn.Conv2d(cfg[5], 16, 1, 1, bias=head_bias)
        self.scales = nn.ModuleList([Scale(init_value=1.0)])
        self.relu = nn.ReLU(inplace=True)
        self.gt_encode = gt_encode
        self.bypass = bypass
        self.num_anchor = num_anchor
        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU(inplace=True),
                
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU(inplace=True),
            )

        if focus_type == 'maxpool':
            self.model_p1 = nn.Sequential(
                # conv_dw(in_channel, self.cfg[0], 1),    #dw1
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(in_channel, self.cfg[0], 1),
                conv_dw(self.cfg[0], self.cfg[1], 1),   #dw2
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[1], self.cfg[2], 1),   #dw3
            )
        elif focus_type == 'v5':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0],
                      kernel_size=1, normalize=normalize, act_fn=act_fn),
                conv_dw(self.cfg[0], self.cfg[1], 1),   #dw2
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[1], self.cfg[2], 1),   #dw3
            )
        elif focus_type == 'v5_double':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0], block_size=4,
                      kernel_size=1, normalize=normalize, act_fn=act_fn),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        elif focus_type == 'v5_v5':
            self.model_p1 = nn.Sequential(
                Focus(in_channel, self.cfg[0],
                      kernel_size=1, normalize=normalize, act_fn=act_fn, num_down=2),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        elif focus_type == 'v5_maxpool':
            self.model_p1 = nn.Sequential(
                FocusMaxPool(in_channel, self.cfg[0],
                             kernel_size=1, normalize=normalize, act_fn=act_fn),
                conv_dw(self.cfg[0], self.cfg[1], 1),
                conv_dw(self.cfg[1], self.cfg[2], 1),
            )
        else:
            self.model_p1 = nn.Sequential(
                conv_dw(in_channel, self.cfg[0], 1),    #dw1
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[0], self.cfg[1], 1),   #dw2
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[1], self.cfg[2], 1),   #dw3
            )

        if out_strides[0] == 8:
            self.model_p2 = nn.Sequential(    
                nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[2], self.cfg[3], 1),   #dw4
                conv_dw(self.cfg[3], self.cfg[4], 1),   #dw5
            )
        elif out_strides[0] == 4:
            self.model_p2 = nn.Sequential(
                # nn.MaxPool2d(kernel_size=2, stride=2),
                conv_dw(self.cfg[2], self.cfg[3], 1),   #dw4
                conv_dw(self.cfg[3], self.cfg[4], 1),   #dw5
            )
        if self.bypass:
            self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
                conv_dw(self.cfg[2] * 4 + self.cfg[4], self.cfg[5], 1),
            )
        else:
            self.model_p3 = nn.Sequential(  #cat dw3(ch:192 -> 768) and dw5(ch:512)
                conv_dw(self.cfg[4], self.cfg[5], 1),
            )
        self.initialize_weights()
        self.out_strides = out_strides
        # init_bias_focal(self.score, class_activation, init_prior, num_classes)

    def get_outstrides(self):
        return torch.tensor(self.out_strides, dtype=torch.int)

    def forward(self, input):
        x = input['image']
        x_p1 = self.model_p1(x)
        x_p1_reorg = self.reorg(x_p1)
        x_p2 = self.model_p2(x_p1)
        if self.bypass:
            x_p3_in = torch.cat([x_p1_reorg, x_p2], 1)
        else:
            x_p3_in = x_p2
        x = self.model_p3(x_p3_in)
        if self.gt_encode:
            out = self.head(x)
        else:
            out = self.head(x)
        # import numpy as np
        # np.save('dir_debug/net_out.npy', out.cpu().detach().numpy())
        score = out[:, :self.num_anchor, :, :]
        # bbox = self.relu(out[:, 2:10, :, :]) 
        bbox = out[:, self.num_anchor:(self.num_anchor * 5), :, :]
        # if self.gt_encode:
            # bbox = self.bbox(x)
        # else:
            # bbox = self.relu(self.scales[0](self.bbox(x)))
        # score = self.score(x)
        return {'features': [[score, bbox]], 'strides': self.get_outstrides()}

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out')
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)  