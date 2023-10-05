import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath



class TransformerStem(nn.Module):
    def __init__(self, in_channels=1, out_channels=16):
        super().__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels//2, kernel_size=10, stride=2, bias=False, padding=4)
        self.act1 = nn.GELU()
        self.bn1 =  nn.BatchNorm1d(out_channels//2)

        self.conv2 = nn.Conv1d(out_channels//2, out_channels, kernel_size=3, stride=1, bias=False,padding=1)
        self.act2 = nn.GELU()
        self.bn2 =  nn.BatchNorm1d(out_channels)

        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=2, bias=False,padding=1)
        self.act3 = nn.GELU()
        self.bn3 =  nn.BatchNorm1d(out_channels)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.bn1(self.act1(self.conv1(x)))
        x = self.bn2(self.act2(self.conv2(x)))
        x = self.bn3(self.act3(self.conv3(x)))

        return x

def _make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, _make_divisible(inp // reduction, 8)),
                nn.SiLU(),
                nn.Linear(_make_divisible(inp // reduction, 8), oup),
                nn.Sigmoid()
        )
 
    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


class MBConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio, fused):
        super(MBConv, self).__init__()
        assert stride in [1, 2]
        hidden_dim = round(in_channels * expand_ratio)
        self.identity = stride == 1 and in_channels == out_channels
        if fused:
            self.conv = nn.Sequential(
                # Fused-MBConv
                nn.Conv1d(in_channels, hidden_dim, 3, stride, 1, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                SELayer(in_channels, hidden_dim),
                # pw-linear
                nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
             self.conv = nn.Sequential(
                nn.Conv1d(in_channels, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                nn.Conv1d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm1d(hidden_dim),
                nn.SiLU(),
                SELayer(in_channels, hidden_dim),
                nn.Conv1d(hidden_dim, out_channels, 1, 1, 0, bias=False),
                nn.BatchNorm1d(out_channels),
            )
 
    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvEmbedding(nn.Module):
    def __init__(self,in_channels,out_channels,depths,stage=1):
        super(ConvEmbedding, self).__init__()
        self.stage = stage
        
        self.mbconvs = nn.ModuleList()
        if self.stage == 1:
            self.mbconvs.append(MBConv(in_channels,out_channels,stride=2, expand_ratio=2, fused=True))
            self.mbconvs.append(MBConv(out_channels,out_channels,stride=1, expand_ratio=2, fused=True))
        else:
            self.mbconvs.append(MBConv(in_channels,out_channels,stride=2, expand_ratio=2, fused=False))
            for _ in range(depths-1):
                self.mbconvs.append(MBConv(out_channels,out_channels,stride=1, expand_ratio=2, fused=False))

        self.norm = nn.LayerNorm(out_channels)
        self.proj = nn.Conv1d(out_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        for _, mbconvs in enumerate(self.mbconvs):
            x = mbconvs(x)
        x = self.norm(self.proj(x).transpose(1,2)).transpose(1,2)
    
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


def conv_separable(in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm1d(in_channels),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm1d(out_channels)
        )


class DWCONV(nn.Module):

    def __init__(self, in_channels, out_channels, stride = 1):
        super(DWCONV, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size = 3,
            stride = stride, padding = 1, groups = in_channels, bias = False)
        self.gelu1 = nn.ReLU() 
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv1(x)
        x = self.gelu1(x)
        result = self.bn1(x)
        return result

class LIL(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(LIL, self).__init__()
        self.DWConv = DWCONV(in_channels, out_channels)

    def forward(self, x):
        result = self.DWConv(x) + x
        return result

class RFFN(nn.Module):

    def __init__(self, in_channels, R):
        super(RFFN, self).__init__()
        exp_channels = int(in_channels * R)
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, exp_channels, kernel_size = 1),
            nn.BatchNorm1d(exp_channels),
            nn.ReLU()
        )

        self.dwconv = nn.Sequential(
            DWCONV(exp_channels, exp_channels),
            nn.BatchNorm1d(exp_channels),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(exp_channels, in_channels, 1),
            nn.BatchNorm1d(in_channels)
        )

    def forward(self, x):
        result = x + self.conv2(self.dwconv(self.conv1(x)))
        return result


class DANE(nn.Module):
    def __init__(self, channel, reduction=16):
        super(DANE, self).__init__()
        self.channel = channel
        self.fc_spatial = nn.Sequential(
            nn.LayerNorm(channel),
            nn.Linear(channel, 1, bias=False),
        )
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc_channel = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.LayerNorm(channel//reduction),
            nn.Linear(channel // reduction, channel, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_channel,x_spatial):
        #B L C
        x_spatial_mask = self.fc_spatial(x_spatial.transpose(1,2)).transpose(1,2) # B 1 L
        x_channel_mask = self.fc_channel(self.avg_pool(x_channel).transpose(1,2)).transpose(1,2) # B C 1
        x_mask = self.sigmoid(x_spatial_mask.expand_as(x_spatial) + x_channel_mask.expand_as(x_spatial))
        return x_spatial * x_mask + x_channel * (1 - x_mask)



class ConvBlock(nn.Module):

    def __init__(self, inplanes, stride, groups=1,norm_layer=nn.BatchNorm1d):
        super(ConvBlock, self).__init__()
        self.inplanes=inplanes
        self.stride = stride
        self.conv1x1_1 = nn.Sequential(nn.Conv1d(inplanes,inplanes,kernel_size=1,stride=1,padding=0,groups=groups,bias=False),
                                        norm_layer(inplanes),
                                        nn.SiLU(inplace=True),
                                        nn.Conv1d(inplanes,inplanes,kernel_size=3,stride=1,padding=1,groups=inplanes,bias=False),
                                        norm_layer(inplanes),
                                        nn.SiLU(inplace=True),
                                        nn.Conv1d(inplanes,inplanes,kernel_size=3,stride=1,padding=1,groups=inplanes,bias=False),
                                        norm_layer(inplanes),
                                        nn.SiLU(inplace=True)
                                        )

        self.conv1 = nn.Sequential(nn.Conv1d(inplanes,inplanes,kernel_size=3,stride=stride,padding=1,groups=inplanes,bias=False),
                                    norm_layer(inplanes),
                                    nn.SiLU(inplace=True)
                                    )
        self.conv1x1_2 = nn.Sequential(nn.Conv1d(inplanes,inplanes,kernel_size=1,stride=1,padding=0,groups=groups,bias=False),
                                        norm_layer(inplanes),
                                        nn.SiLU(inplace=True)
                                        )

    def forward(self, x):
        out = self.conv1x1_1(x)
        x_out = out
        out = self.conv1(out)
        out = self.conv1x1_2(out)
        return x_out,out


class SMHSA(nn.Module):

    def __init__(self, channels, d_k, d_v, stride, heads, dropout,qkv_bias=False,attn_drop=0., proj_drop=0.):
        super(SMHSA, self).__init__()
        self.dwconv_k = DWCONV(channels, channels, stride = stride)
        self.dwconv_v = DWCONV(channels, channels, stride = stride)
        self.fc_q = nn.Linear(channels, heads * d_k, bias=qkv_bias)
        self.fc_k = nn.Linear(channels, heads * d_k, bias=qkv_bias)
        self.fc_v = nn.Linear(channels, heads * d_v, bias=qkv_bias)
        self.fc_o = nn.Linear(heads * d_k, channels)

        self.channels = channels
        self.d_k = d_k
        self.d_v = d_v
        self.stride = stride
        self.heads = heads
        self.dropout = dropout
        self.scaled_factor = self.d_k ** -0.5
        self.num_patches = (self.d_k // self.stride) ** 2

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        b, c, l = x.shape

        x_reshape = x.permute(0, 2, 1) 

        # Get q, k, v
        q = self.fc_q(x_reshape)
        q = q.view(b, l, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  

        k = self.dwconv_k(x)
        k_b, k_c, k_l = k.shape
        k = k.view(k_b, k_c, k_l).permute(0, 2, 1).contiguous()
        k = self.fc_k(k)
        k = k.view(k_b, k_l, self.heads, self.d_k).permute(0, 2, 1, 3).contiguous()  

        v = self.dwconv_v(x)
        v_b, v_c, v_l = v.shape
        v = v.view(v_b, v_c, v_l).permute(0, 2, 1).contiguous()
        v = self.fc_v(v)
        v = v.view(v_b, v_l, self.heads, self.d_v).permute(0, 2, 1, 3).contiguous() 

        attn = torch.einsum('... i d, ... j d -> ... i j', q, k) * self.scaled_factor
        attn = torch.softmax(attn, dim = -1) 

        attn = self.attn_drop(attn)

        result = torch.matmul(attn, v).permute(0, 2, 1, 3)
        result = result.contiguous().view(b, l, self.heads * self.d_v)
        result = self.fc_o(result).view(b, self.channels, -1)
        result = self.proj_drop(result)
        return result



class MyFormerBlock(nn.Module):
    def __init__(self,dim,d_k,num_heads, stride,mlp_ratio=4., 
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm,R=1):
        super(MyFormerBlock,self).__init__()  

        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        
        self.norm1 = norm_layer(dim)
        self.attn = SMHSA(dim, d_k, d_k, stride, num_heads, 0.0, qkv_bias=qkv_bias,attn_drop=attn_drop, proj_drop=drop)

        self.ffn = RFFN(dim, R)

        self.select = DANE(channel = dim)

        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)

    def forward(self,x,x_downsample):
        x = self.norm1(x.transpose(1,2)).transpose(1,2)
        x = self.attn(x)
        x = self.ffn(x)
        x = self.select(x, x_downsample)
        x = x + self.drop_path(self.mlp(self.norm2(x.transpose(1,2)))).transpose(1,2)
        return x


class BasicLayer(nn.Module):
    def __init__(self, dim, d_k, depth, num_heads,stride,mlp_ratio=4., qkv_bias=True, drop=0., attn_drop=0.,
                 drop_path=0.,norm_layer=nn.LayerNorm,pe=True):
        super(BasicLayer,self).__init__()    
        self.dim = dim
        self.depth = depth
        self.pe = pe

        self.convlayer = ConvBlock(inplanes=dim,stride = 2)
        self.maxpool = nn.MaxPool1d(kernel_size = 3, stride = 2, padding = 1)
        self.multiresolution_con = conv_separable(2 * dim, dim, 1)

        if self.pe:
            self.lil = LIL(self.dim, self.dim)

        # build transformer encoders
        self.blocks = nn.ModuleList([
                    MyFormerBlock(dim=dim, d_k=d_k,
                                 num_heads=num_heads,
                                 stride=stride, 
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, 
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
                    for i in range(depth)])


    def forward(self,x):
        x_spatial, x_inter = self.convlayer(x)
        x_pool = self.maxpool(x)

        L = x_spatial.shape[2]
        pad_input = (L % 2 == 1)
        if pad_input:
            x_spatial = F.pad(x_spatial, (0, L % 2))

        x0 = x_spatial[:, :, 0::2] 
        x1 = x_spatial[:, :, 1::2]  
        x_spatial = torch.cat([x0, x1], 1)  
        x_spatial = self.multiresolution_con(x_spatial) 
        x = x_pool + x_inter

        if self.pe:
            x = self.lil(x)

        for blk in self.blocks:
            x = blk(x, x_spatial)

        return x



class IEEGHCT(nn.Module):
    def __init__(self,in_channels=1,
                num_classes=3,  # The number of classes for recognition.
                ce_depths=4, # MBConvs numbers of Convolutional Embeddind 
                embed_dim=8,
                d_k=32,
                num_heads=[1, 2, 4, 8], # The number of heads in different stages.
                strides = [4, 4, 2, 2], # SMHSA
                depths=[1, 2, 4, 8],  # The number of blocks in each stage.
                mlp_ratio=4, # The MLP expansion rate.
                qkv_bias=False,  # Whether adding bias to qkv.
                drop_rate=0.,  # Dropout rate.
                attn_drop_rate=0.,  # Dropout rate on attention values.
                norm_layer=nn.LayerNorm,  # The norm layer.
                pe=True, # Positional Embedding, LIL
                ):
        super(IEEGHCT,self).__init__()

        self.num_classes = num_classes
        self.depths = depths
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.pe = pe
        self.mlp_ratio = mlp_ratio

        self.pos_drop = nn.Dropout(p=drop_rate)

        # Stem layer
        self.stem = TransformerStem(in_channels, embed_dim)

        # CE Blocks
        self.conv_embeds = nn.ModuleList()
        for i in range(len(depths)):
            self.conv_embeds.append(ConvEmbedding(embed_dim * 2 ** i,embed_dim * 2 ** (i+1), depths=ce_depths,stage=i+1))

        # Transformer Blocks
        self.layers = nn.ModuleList()
        for k in range(self.num_layers):
            layer = BasicLayer(dim=embed_dim * 2 ** (k+1),d_k=d_k,
                                depth=depths[k],
                                num_heads=num_heads[k],
                                stride=strides[k],
                                mlp_ratio=mlp_ratio,
                                qkv_bias=qkv_bias,
                                drop=drop_rate,
                                attn_drop=attn_drop_rate,
                                norm_layer=norm_layer
                                )
            self.layers.append(layer)

        self.norm = norm_layer(embed_dim * 2 ** len(depths))
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(embed_dim * 2 ** len(depths), num_classes) if num_classes > 0 else nn.Identity()


    def forward_features(self, x):
        x = self.stem(x)
        x = self.pos_drop(x)

        for i in range(self.num_layers):
            x = self.conv_embeds[i](x)
            x = self.layers[i](x)

        x = self.norm(x.transpose(1,2)).transpose(1,2)
        return x  


    def forward(self, x):
        x = self.forward_features(x)  
        x = self.avgpool(x)  
        x = torch.flatten(x, 1)
        x = self.head(x)

        return x








if __name__ == '__main__':
    x = torch.rand((32,1,15000))
    model = IEEGHCT(depths=[1, 2, 4, 2])
    y = model(x)
