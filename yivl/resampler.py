import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
from transformers.activations import ACT2FN

# import open_clip


class FFN(nn.Module):
    def __init__(self, embed_dim, ff_dim, output_dim):
        super().__init__()
        self.linear_in = nn.Linear(embed_dim, ff_dim, bias=False)
        self.linear_out = nn.Linear(ff_dim, output_dim, bias=False)
        self.act = ACT2FN["gelu_new"]

    def forward(self, hidden_states):
        hidden_states = self.act(self.linear_in(hidden_states))
        hidden_states = self.linear_out(hidden_states)
        return hidden_states


class SelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, drop_out_rate=0):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_out_rate)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, hidden_states, attn_mask=None):
        normed_hidden_states = self.layer_norm(hidden_states)

        query = self.q_proj(normed_hidden_states).permute(1, 0, 2)
        key = self.k_proj(normed_hidden_states).permute(1, 0, 2)
        value = self.v_proj(normed_hidden_states).permute(1, 0, 2)

        # Compute multi-head self-attention
        attn_output, _ = self.multihead_attn(query, key, value)

        # Reshape back to original shape
        attn_output = attn_output.permute(1, 0, 2)

        # Apply linear layer and dropout, add residual connection
        attn_output = hidden_states + self.dropout(self.linear(attn_output))

        return attn_output


class CrossAttention(nn.Module):
    def __init__(self, kv_dim, embed_dim, num_heads, drop_out_rate=0):
        super().__init__()
        self.num_heads = num_heads
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, embed_dim, bias=False)

        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)
        self.linear = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(drop_out_rate)

        self.layer_norm = nn.LayerNorm(embed_dim)
        self.ln_kv = nn.LayerNorm(kv_dim)

    def forward(self, x, hidden_states, attn_mask=None, add_residual=False):
        normed_hidden_states = self.layer_norm(hidden_states)
        query = self.q_proj(normed_hidden_states).permute(1, 0, 2)

        x = self.ln_kv(x)
        key = self.k_proj(x).permute(1, 0, 2)
        value = self.v_proj(x).permute(1, 0, 2)

        # Compute multi-head self-attention
        attn_output, _ = self.multihead_attn(query, key, value, attn_mask=attn_mask)

        # Reshape back to original shape
        attn_output = attn_output.permute(1, 0, 2)

        if add_residual:
            # Apply linear layer and dropout, add residual connection
            attn_output = hidden_states + self.dropout(self.linear(attn_output))
        else:
            attn_output = self.dropout(self.linear(attn_output))

        return attn_output


class Resampler(nn.Module):
    """
    A 2D perceiver-resampler network with one cross attention layers by
        (grid_size**2) learnable queries and 2d sincos pos_emb
    Outputs:
        A tensor with the shape of (grid_size**2, embed_dim)
    """

    def __init__(
        self,
        num_queries,
        embed_dim,
        num_heads,
        kv_dim,
        ff_dim,
        output_dim,
        norm_layer=nn.LayerNorm,
        has_self_attn=False,
        num_patches=None,
    ):
        super().__init__()
        self.num_queries = num_queries
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        if type(self.num_queries) is list:
            self.query = nn.Parameter(
                torch.zeros(max(self.num_queries), self.embed_dim)
            )
            self.num_patches = num_patches
        else:
            self.query = nn.Parameter(torch.zeros(self.num_queries, self.embed_dim))
        trunc_normal_(self.query, std=0.02)

        self.cross_attn = CrossAttention(kv_dim, embed_dim, num_heads)

        self.has_self_attn = has_self_attn
        if has_self_attn:
            self.self_attn = SelfAttention(embed_dim, num_heads)

        self.ln_ffn = norm_layer(embed_dim)
        self.ffn = FFN(embed_dim, ff_dim, output_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x, attn_mask=None):
        bs = x.shape[0]

        queries = self.query.unsqueeze(0).repeat(bs, 1, 1)
        if type(self.num_queries) is list and x.shape[1] < self.num_patches:
            queries = queries[:, : min(self.num_queries), :]

        if attn_mask is not None:
            attn_mask = attn_mask.repeat_interleave(self.num_heads, 0)
            attn_mask = attn_mask.unsqueeze(1).expand(-1, queries.size(1), -1)

        attention_out = self.cross_attn(x, queries, attn_mask=attn_mask)

        if self.has_self_attn:
            attention_out = self.self_attn(attention_out)

        out = self.ffn(self.ln_ffn(attention_out))

        return out
