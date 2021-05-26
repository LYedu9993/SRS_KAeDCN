import torch
import torch.nn as nn
import torch.nn.functional as F
from Model_class.CSAM import *


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def LayerNorm(normalized_shape, eps=1e-5, elementwise_affine=True):
    if torch.cuda.is_available():
        try:
            from apex.normalization import FusedLayerNorm
            return FusedLayerNorm(normalized_shape, eps, elementwise_affine)
        except ImportError:
            pass
    return torch.nn.LayerNorm(normalized_shape, eps, elementwise_affine)


def Linear(in_features, out_features, bias=True):
    m = nn.Linear(in_features, out_features, bias)
    nn.init.xavier_uniform_(m.weight)
    if bias:
        nn.init.constant_(m.bias, 0.)
    return m


def unfold1d(x, kernel_size, padding_l, pad_value=0):  
    if kernel_size > 1:
        T, B, C = x.size()  
        x = F.pad(x, (0, 0, 0, 0, padding_l, kernel_size - 1 - padding_l), value=pad_value)
        x = x.as_strided((T, B, C, kernel_size), (B * C, C, 1, B * C))  
    else:
        x = x.unsqueeze(3)
    return x


class DynamicConv1dTBC(nn.Module):
    def __init__(self,
                 args,
                 input_size,
                 kernel_size=1,
                 padding_l=None,
                 num_heads=1,
                 unfold=False,
                 weight_dropout=0.,
                 weight_softmax=False,
                 renorm_padding=False,
                 bias=False,
                 conv_bias=False,
                 query_size=None):

        super().__init__()
        self.args = args
        self.input_size = input_size
        self.query_size = input_size if query_size is None else query_size  
        self.kernel_size = kernel_size
        self.padding_l = padding_l
        self.num_heads = num_heads
        self.weight_dropout = weight_dropout
        self.weight_softmax = weight_softmax
        self.renorm_padding = renorm_padding
        self.unfold = unfold
        self.weight_linear = Linear(self.query_size, num_heads * kernel_size * 1, bias=bias)

        if conv_bias:
            self.conv_bias = nn.Parameter(torch.Tensor(input_size))
        else:
            self.conv_bias = None
        self.reset_parameters()  

        if args.isKG:
            self.CSAM_Module = CSAM(args=self.args, RS_dim=args.embed_dim, KG_dim=args.entity_dim, _kernel_size=args.spatial_kernel_size, _reduction=args.channel_reduction)
        else:
            self.CSAM_Module = CSAM(args=self.args, RS_dim=args.embed_dim, _kernel_size=args.spatial_kernel_size, _reduction=args.channel_reduction)

    @property
    def in_proj(self):
        return self.weight_linear.out_features == self.input_size + self.num_heads * self.kernel_size

    def reset_parameters(self):  
        self.weight_linear.reset_parameters()
        if self.conv_bias is not None:
            nn.init.constant_(self.conv_bias, 0.)

    def forward(self, x, x_kg=None, incremental_state=None, query=None):
       

        query = x  

        
        if self.args.isKG:  
            output = self._forward_unfolded(x=x, incremental_state=incremental_state, query=query, x_kg=x_kg)
        else:
            output = self._forward_unfolded(x, incremental_state, query)

        if self.conv_bias is not None:
            output = output + self.conv_bias.view(1, 1, -1)

        return output

    def _forward_unfolded(self, x, incremental_state, query, x_kg=None):

        T, B, C = x.size()
        K, H = self.kernel_size, self.num_heads
        R = C // H  

        assert R * H == C == self.input_size

        weight = self.weight_linear(query).view(T * B * H, -1)  

        weight = F.dropout(weight, self.weight_dropout, training=self.training, inplace=False)
        
        padding_l = self.padding_l
        if K > T and padding_l == K - 1:
            weight = weight.narrow(1, K - T, T)
            K, padding_l = T, T - 1

        
        x_unfold = unfold1d(x, K, padding_l, 0)  

        x_CSAM_input = x_unfold.transpose(3, 2)  
        x_CSAM_input_reshape = x_CSAM_input.reshape((T * B, K, C))  

        if self.args.isKG:
            x_kg_unfold = unfold1d(x_kg, K, padding_l, 0)
            x_kg_CSAM_input = x_kg_unfold.transpose(3, 2) 
            x_kg_CSAM_input_reshape = x_kg_CSAM_input.reshape((T * B, K, self.args.entity_dim))  
            x_out = self.CSAM_Module(x=x_CSAM_input_reshape, x_kg=x_kg_CSAM_input_reshape)   
        else:
            x_out = self.CSAM_Module(x=x_CSAM_input_reshape) 

        back_reshape_x_out = x_out.reshape((T, B, K, C))
        back_transpose_x_out = back_reshape_x_out.transpose(3, 2)  
        x_unfold = back_transpose_x_out.view(T * B * H, R, K)  
        if self.weight_softmax:
            weight = F.softmax(weight, dim=1)

        weight = weight.narrow(1, 0, K)

        
        output = torch.bmm(x_unfold, weight.unsqueeze(2))  
       
        output = output.view(T, B, C)
        return output

