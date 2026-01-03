# implementing the building blocks for building a transformer from sratch
import torch
import torch.nn as nn 
import torch.nn.functional as F 
from copy import deepcopy

# the positional encodings layer
class PositionalEmbedding(nn.Module):
    def __init__(self, max_length, embed_dim, dropout=0.1):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.randn(max_length, embed_dim)*0.02)
        self.dropout = nn.Dropout(dropout)
    def forward(self, X):
        # assuming input is batch first and of shape: (B, seq_len, dim)
        return self.dropout(X + self.pos_embed[:X.size(1)])

# the multihead attention layer
class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super().__init__()
        self.h = num_heads
        self.d = embed_dim // num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
    
    def split_heads(self, X):
        return X.view(X.size(0), X.size(1), self.h, self.d).transpose(1,2)
    
    def forward(self, query, key, value, attn_mask=None):
        q = self.split_heads(self.q_proj(query)) # B,h, Lq, dq
        k = self.split_heads(self.k_proj(key)) # B,h,Lk,dk
        v = self.split_heads(self.v_proj(value)) # B,h,Lv,dv
        scores = q @ k.transpose(2,3)/self.d**0.5 # scaled dot product attention

        # masking support
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask, -torch.inf)
        
        weights = scores.softmax(dim=-1)
        Z = self.dropout(weights) @ v # B,h,Lq,dv
        Z = Z.transpose(1,2) # B,Lq,h,dv
        Z = Z.reshape(Z.size(0),Z.size(1),self.h*self.d)
        return self.out_proj(Z) # B,Lq,embed_dim

# the transformer encoder layer:
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, src, src_mask=None):
        attn = self.self_attn(src,src,src,attn_mask=src_mask) # out shape: (B, Lq, embed_dim)
        Z = self.norm1(src + self.dropout(attn)) # out shape: (B, L, embed_dim)
        ff = self.dropout(\
                        self.linear2(\
                            self.dropout(\
                                self.linear1(Z).relu()
                                ))) # project then compress so embed_dim goes to 2048 first then back to embed_dim
        return self.norm2(Z + ff)

# the transformer encoder layer
class TransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = nn.ModuleList([deepcopy(encoder_layer) for _ in range(num_layers)])
        self.norm = norm
    
    def forward(self, src, mask=None):
        Z = src 
        for layer in self.layers:
            Z = layer(Z, mask)
        if self.norm is not None:
            Z = self.norm(Z)
        return Z

# now the transformer we'll be implementing for timeseries
class Transformer(nn.Module):
    def __init__(self, d_model=512, nheads=8, num_encoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        encoder_layer = TransformerEncoderLayer(d_model, nheads, dim_feedforward, dropout)
        norm1 = nn.LayerNorm(d_model)
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, norm1)
    
    def forward(self, src, src_mask):
        return self.encoder(src, src_mask)