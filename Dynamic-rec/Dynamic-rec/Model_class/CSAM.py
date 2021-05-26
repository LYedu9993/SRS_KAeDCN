import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttentionModule(nn.Module):


    def __init__(self, dim, reduction=1):
        super(ChannelAttentionModule, self).__init__()
        mid_dim = dim // reduction
        self.BottleNeck = nn.Sequential(
            nn.Linear(in_features=dim, out_features=dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=dim, out_features=dim)
        )

        self.sigmoid = nn.Sigmoid()

        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=1,
                              kernel_size=7,
                              stride=1,
                              padding=3)

    def forward(self, x): 


        out = self.BottleNeck(x)

        return self.sigmoid(out)


class SpatialAttentionModule(nn.Module):
    def __init__(self, k_size = 7):
        super(SpatialAttentionModule, self).__init__()
        _kernel_size = k_size
        _padding = (_kernel_size - 1) // 2
        _outchannel = 1
        self.conv1d = nn.Conv1d(in_channels=2, out_channels=_outchannel,
                                kernel_size=_kernel_size,
                                stride=1,
                                padding=_padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avgout = torch.mean(x, dim=-1, keepdim=True)    
        maxout, _ = torch.max(x, dim=-1, keepdim=True)  
        out = torch.cat([avgout, maxout], dim=-1)       
        out = out.permute(0, 2, 1)                      
        out_conv = self.conv1d(out)
        
        out_f = self.sigmoid(out_conv)                       
        out_f = out_f.permute(0, 2, 1)                      
        return out_f


class CSAM(nn.Module):
    def __init__(self, args, RS_dim, KG_dim=None, _kernel_size=7, _reduction=16):
        super(CSAM, self).__init__()
        self.args = args
        self.channel_attention = ChannelAttentionModule(RS_dim, _reduction)
        self.spatial_attention = SpatialAttentionModule(_kernel_size)

        self.Liner_KG_RS = nn.Linear(in_features=KG_dim, out_features=RS_dim)

        if KG_dim != None:
            self.channel_attention_KG = ChannelAttentionModule(KG_dim, _reduction)
            self.spatial_attention_KG = SpatialAttentionModule(_kernel_size)

            _padding = (_kernel_size - 1) // 2
            _outchannel = 1
            self.G_conv1d = nn.Conv1d(in_channels=RS_dim, out_channels=_outchannel,
                                 kernel_size=_kernel_size,
                                 stride=1,
                                 padding=_padding)
            self.G_Sigmod = nn.Sigmoid()
            self.G_Relu = nn.ReLU()

    def Generate_gate(self, x):
        x_input = x.permute(0, 2, 1)                  
        out = self.G_Sigmod(self.G_Relu(self.G_conv1d(x_input)))  
        _out = 1 - out
        out = out.permute(0, 2, 1)
        _out = _out.permute(0, 2, 1)
        return out, _out

    def forward(self, x, x_kg=None):
        if self.args.isKG:
            residual = x
            x_inial = x
            
            rate_RS, rate_KG = self.Generate_gate(x_inial)

            out_RS = self.channel_attention(x) * x
            out_RS = self.spatial_attention(out_RS) * out_RS

            out_KG = self.channel_attention_KG(x_kg) * x_kg
            out_KG = self.spatial_attention_KG(out_KG) * out_KG
            out_KG = self.Liner_KG_RS(out_KG)  

            out = rate_RS * out_RS + rate_KG * out_KG + residual
        else:
            residual = x
            out = self.channel_attention(x) * x
            out = self.spatial_attention(out) * out
            out = out + residual

        return out