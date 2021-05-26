import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np
sys.path.append("..")
from Model_class.modules_new import *
from Model_class.CSAM import *




class ConvRec(nn.Module):

    def __init__(self, args, itemnum):
        super(ConvRec, self).__init__()

        add_args(args)

        self.args = args  # Save parameters
        self.dropout = args.dropout
        self.maxlen = args.maxlen
        self.itemnum = itemnum

        self.KGE_path = args.KGE_path
        self.KGE = args.KGE
        self.entity_dim = args.entity_dim
        self.KGtype = args.KGtype

        self.item_embedding = Embedding(itemnum + 1, args.embed_dim, 0)

        # Loading the embedded representation of an entity --------------------------
        entity_embs = np.load(self.KGE_path + self.KGE + '_' + str(self.entity_dim) + '_' + self.KGtype + '.npy')
        shape_entity_embs = entity_embs.shape  # Calculate the latitude of all entity vectors.
        # Loading entity vectors (from TransE training)--------------------------
        if shape_entity_embs[0] != itemnum + 1:
            print('Error，shape_entity_embs[0] != itemnum + 1')
            sys.exit(1)
        self.entity_embeddings = nn.Embedding(shape_entity_embs[0], self.entity_dim)
        self.entity_embeddings.weight.data.copy_(torch.from_numpy(entity_embs))
        self.entity_embeddings.weight.requires_grad = False
        # loading --------------------------

        self.embed_scale = math.sqrt(args.embed_dim)
        self.position_encoding = Embedding(args.maxlen, args.embed_dim, 0)

        self.layers = nn.ModuleList([])  # Layers list
        self.layers.extend([
            ConvRecLayer(args, kernel_size=args.decoder_kernel_size_list[i])
            for i in range(args.layers)
        ])

        self.layer_norm = LayerNorm(args.embed_dim)




    def forward(self, seq, pos=None, neg=None, test_item=None):
        """

        1. Training: loss, _ = conv_model.forward(seq, pos=pos)

        2.Testing： _, rank_20 = model.forward(seq, test_item = all_items_tensor)

        """

        x = self.item_embedding(seq)
        x_kg = self.entity_embeddings(seq)





        # B x T x C -> T x B x C
        x = x.transpose(0, 1)
        x_kg = x_kg.transpose(0, 1)

        attn = None  # useless

        inner_states = [x]

        # decoder layers
        for layer in self.layers:   # The number of layers.
            x = self.layer_norm(x)  # norm
            if self.args.isKG:      # Whether using KG.
                x, attn = layer(x=x, x_kg=x_kg)
            else:
                x, attn = layer(x=x)
            inner_states.append(x)

        # if self.normalize:
        x = self.layer_norm(x)  # Normalization layer

        # T x B x C -> B x T x C
        x = x.transpose(0, 1)
        # x.size = (batch_size , maxlen-1, dim_item)

        seq_emb = x.contiguous().view(-1, x.size(-1))
        # reshaping it to [arg.batch_size x args.maxlen * args.hidden_units]
        # seq_emb.shape = 128 * 30-1  ,  200  [batch_size * max_len -1 ,  dim_item]
        pos_logits = None
        neg_logits = None
        rank_20 = None
        istarget = None
        loss = None

        # This function is called only during training.
        if pos is not None:
            # pos ,shape = [128, 29]
            pos = torch.reshape(pos, (-1,))  # shape = [batch_size * max_len -1]

            nnz = torch.ne(pos, 0).nonzero().squeeze(-1)

            neg = torch.randint(1, self.itemnum + 1,  # Sampling
                                (self.args.num_neg_samples, nnz.size(0)),
                                # 400 128*29
                                device=self.args.computing_device)
            neg = neg.to(torch.int64)
            pos_emb = self.item_embedding(pos[nnz])  # 1.Positive Example （128 * 29,    200）
            neg_emb = self.item_embedding(neg)  # 2.Negative Example （400, 128*29, 200）
            seq_emb = seq_emb[nnz]  # 3.seq （128*29,      200）


            # sequential context
            pos_logits = torch.sum(pos_emb * seq_emb, -1)
            neg_logits = torch.sum(neg_emb * seq_emb, -1)  
            negative_scores = torch.sum(
                (1 - torch.sigmoid(neg_logits) + 1e-24).log(), dim=0)

            loss = torch.sum(-(torch.sigmoid(pos_logits) + 1e-24).log() -
                             negative_scores) / nnz.size(0)

        # Call this function during testing
        # The dataset used at this point is :valid
        # seq - 1
        # ground_truth: session[-1] - 1
        if test_item is not None:
            """
            _, rank_20 = model.forward(seq, test_item = all_items_tensor)
            """
            test_item_emb = self.item_embedding(test_item)
            # test_item_emb.shape = [10000, 200]

            seq_emb = seq_emb.view(seq.size(0), seq.size(1), -1)
            # seq.size(0) = 128  seq.size(1) = 29
            # seq_emb : [batch_size, max_len -1,  dim_item]   3 dim
            seq_emb = seq_emb[:, -1, :]
            # seq_emb.shape: 128 1 200
            seq_emb = seq_emb.contiguous().view(-1, seq_emb.size(-1))  # [128, 200]
            test_logits = torch.mm(seq_emb, test_item_emb.t())  # check
            # 128 * 200  * 200 * 10000 = 128 * 10000

            test_logits_indices = torch.argsort(-test_logits)
            rank_20 = test_logits_indices[:, :20]
            # Outputs the subscripts of the top 20 highest scoring commodity objects

        return loss, rank_20


class ConvRecLayer(nn.Module):  # Dynamic convolution layer

    def __init__(self, args, kernel_size=0):
        super().__init__()
        self.args = args
        self.embed_dim = args.embed_dim  # 200

        self.conv = DynamicConv1dTBC(
                                     args,
                                     args.embed_dim,  # 200
                                     kernel_size,  # 5
                                     padding_l=kernel_size - 1,  # 4
                                     weight_softmax=args.weight_softmax,  # True default: False
                                     num_heads=args.heads,
                                     unfold=None,
                                     weight_dropout=args.weight_dropout)

        self.dropout = args.dropout
        self.layer_norm = LayerNorm(self.embed_dim)

        self.fc1 = Linear(self.embed_dim, args.ffn_embed_dim)
        self.fc2 = Linear(args.ffn_embed_dim, self.embed_dim)

    def forward(self, x, x_kg=None, conv_mask=None, conv_padding_mask=None):

        T, B, C = x.size()
        if self.args.isKG:
            x = self.conv(x=x, x_kg=x_kg)
        else:
            x = self.conv(x=x)
        x = self.layer_norm(x)
        attn = None
        residual = x
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x += residual
        return x, attn


def add_args(args):
    if len(args.decoder_kernel_size_list) == 1:
        # For safety in case kernel size list does not match with # of convolution layers
        args.decoder_kernel_size_list = args.decoder_kernel_size_list * args.layers

    args.weight_softmax = True

    print("\n Model load/define: Model arguments:\n", args)
    print('\n')