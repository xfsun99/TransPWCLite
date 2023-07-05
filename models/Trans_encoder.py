import torch
import torch.nn as nn
import numpy as np
import copy


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src, pos):
        output = src
        for layer in self.layers:
            output = layer(output, pos)

        if self.norm is not None:
            output = self.norm(output)

        return output


class TransformerEncoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=512, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = nn.ReLU(inplace=True)

    def pos_embed(self, src, pos):
        batch_pos = pos.unsqueeze(1).repeat(1, src.size(1), 1)
        return src + batch_pos

    def forward(self, src, pos):
        # src_mask: Optional[Tensor] = None,
        # src_key_padding_mask: Optional[Tensor] = None):
        # pos: Optional[Tensor] = None):

        q = k = self.pos_embed(src, pos)
        src2 = self.self_attn(q, k, value=src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class Encoder(nn.Module):
    def __init__(self,l):
        super(Encoder, self).__init__()
        maps = 16
        nhead = 4
        dim_feedforward = 512
        dropout = 0.1
        num_layers = 6
        self.L = l
        encoder_layer = TransformerEncoderLayer(
            maps,
            nhead,
            dim_feedforward,
            dropout)
        encoder_norm = nn.LayerNorm(maps)
        self.encoder = TransformerEncoder(encoder_layer, num_layers, encoder_norm)
        self.cls_token = nn.Parameter(torch.randn(1, 1, maps))
        self.up_con = nn.Conv2d(2, 16, kernel_size=(1, 1), stride=(1, 1))
        self.down_con = nn.Conv2d(16, 2, kernel_size=(1, 1), stride=(1, 1))
        self.avg = nn.AvgPool2d
        self.Embedding = nn.Embedding
        self.ConvTranspose2d = nn.ConvTranspose2d
    def forward(self,flow):
        channel_Ori = flow.size(1)
        flow = self.up_con(flow)
        bs = flow.size(0)
        channel = flow.size(1)
        h = h_Ori = flow.size(2)
        w = w_Ori = flow.size(3)
        resize = False
        if h_Ori % 2 != 0:
            h = h_Ori + 1
            resize = True
        if w_Ori % 2 != 0:
            w = w_Ori + 1
            resize = True
        with torch.no_grad():
            flow_ = flow.clone()
            if resize:
                flow_ = flow_.resize_(bs, channel, h, w)
            if self.L <= 2:
                d = 2
            else:
                if self.L == 3:
                    d = 4
                else:
                    d = 8
        flow = nn.AvgPool2d(d, d)(flow_)
        h2 = flow.size(2)
        w2 = flow.size(3)
        flow = flow.flatten(2)
        flow = flow.permute(2, 0, 1)
        cls = self.cls_token.repeat((1, bs, 1))
        flow = torch.cat([cls, flow], 0)
        position = torch.from_numpy(np.arange(0, int(h2 * w2 + 1))).cuda()
        pos_feature = self.Embedding(int(h2 * w2 + 1), 16).cuda()(position)
        flow = self.encoder(flow, pos_feature)
        flow = flow.permute(1, 2, 0)
        with torch.no_grad():
            flow1 = flow.clone().resize_((bs, channel, h2 * w2)).reshape((bs, channel, h2, w2))
            flow = self.down_con(flow1)
            flow = self.ConvTranspose2d(flow.size(1), flow.size(1), kernel_size=d, stride=d).cuda()(flow)
            flow = flow.clone().resize_((bs, channel_Ori, min(h, h_Ori), min(w, w_Ori)))

        return flow



