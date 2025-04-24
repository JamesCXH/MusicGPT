import torch
from torch import nn
from torch.nn import functional as F
import math
from einops import einsum, rearrange


class NewGELU(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


class RotaryPositionalEmbeddings(nn.Module):
    """ 
    Implementation of RoPE introduced in the paper RoFormer: Enhanced Transformer with Rotary Position Embedding.
    Reference: https://arxiv.org/abs/2104.09864
    """

    def __init__(self, d: int, base: int = 10_000):
        super().__init__()

        self.d = d
        self.base = base
        self.cos_cache = torch.empty(0)
        self.sin_cache = torch.empty(0)


    def _build_cache(self, x: torch.Tensor):
        """
        Build a cache for efficient computation of the rotary embeddings.
        """
        #print(f'RoPE input size: {x.size()}')
        b, h, t, n_embd = x.size() # batch, n_heads, seq_len, embed_dim per head

        assert (self.d == n_embd)
        assert (n_embd % 2 == 0)
        
        i = torch.arange(1, self.d//2 + 1, device=x.device)  # i values from 1 to d/2
        Theta = self.base ** (-2 * (i - 1) / self.d) # d/2 theta values
        Theta = Theta.unsqueeze(0) # [1, d/2]

        j = torch.arange(1, t+1, device=x.device).unsqueeze(1) # [t, 1]

        C = j * Theta # [t, d/2]
        C = torch.cat([C, C], dim=1) # [t, d]

        cos_matrix = torch.cos(C) # [t, d]
        sin_matrix = torch.sin(C) # [t, d]

        # need cosine and sine matrix for each head:
        cos_matrix = cos_matrix.unsqueeze(1).expand(-1, h, -1) # [t, h, d]
        sin_matrix = sin_matrix.unsqueeze(1).expand(-1, h, -1) # [t, h, d]

        self.cos_cache = rearrange(cos_matrix, 't h d -> h t d')
        self.sin_cache = rearrange(sin_matrix, 't h d -> h t d')

        # push to device
        self.cos_cache = self.cos_cache.to(x.device)
        self.sin_cache = self.sin_cache.to(x.device)

        #print(f'sin matrix size: {self.sin_cache.size()}')
    

    def forward(self, x: torch.Tensor):
       
        b, h, t, n_embed = x.size()
        
        if self.cos_cache.numel() == 0 or self.cos_cache.size(1) != t:
            self._build_cache(x)
        
        assert((n_embed % 2 == 0))

        first_half = x[:, :, :, :n_embed//2] 
        second_half = x[:, :, :, n_embed//2:]

        cos_matrix = self.cos_cache
        sin_matrix = self.sin_cache

        if self.cos_cache.size(0) != h:
          assert (self.cos_cache.size(0) > h) # cos_cache.size(0) = num_query_heads, h = num_kv_heads
          cos_matrix = self.cos_cache[:h, :, :]
          sin_matrix = self.sin_cache[:h, :, :]

        #print(f'x shape: {x.size()}, cos shape: {self.cos_cache.size()}, sin shape: {self.sin_cache.size()}')
        return (x * cos_matrix) + (torch.concat((-second_half, first_half), dim=-1) * sin_matrix)
        


class MultiHeadAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_query_head == 0
        self.n_head = config.n_query_head
        self.n_embd = config.n_embd
        self.ROPE = None

        self.head_dim = config.n_embd // self.n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)

        # output projection
        self.out = nn.Linear(config.n_embd, config.n_embd)

        # regularization
        # self.attn_dropout = nn.Dropout(config.attn_pdrop)
        self.dropout_p = config.attn_pdrop
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

        self.rope = config.rope
        if self.rope:
            self.ROPE = RotaryPositionalEmbeddings(self.n_embd // self.n_head)


    def forward(self, x, mask=None):

        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3C)
        qkv = qkv.view(B, T, 3, self.n_head, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # each (B, T, H, D)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]  # (B, H, T, D)

        if self.rope:
            q = self.ROPE(q)
            k = self.ROPE(k)

        if mask is not None:
            assert mask.dim() == 4 \
                   and mask.shape[0] == 1 \
                   and mask.shape[1] == 1 \
                   and mask.shape[2] == T \
                   and mask.shape[3] == T, \
                f"mask must have shape (1, 1, {T}, {T}); got {tuple(mask.shape)}"
            attn_mask = mask.bool()  # True = block
        else:
            attn_mask = None

        # attn_mask = mask.bool() if mask is not None else None

        ctx = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p,
        )

        # Flash kernel doesn’t return weights; return None to keep signature.
        ctx = ctx.transpose(1, 2).contiguous().view(B, T, -1)  # (B, T, C)
        out = self.resid_dropout(self.out(ctx))
        return out, None

# class GroupedQueryAttention(nn.Module):
#     """
#     An implementation of group query attention. Refer to the CausalSelfAttention class to structure your implementation.
#     """
#
#     def __init__(self, config):
#         super().__init__()
#
#         """
#         Implementation of grouped query attention
#         """
#
#         assert config.n_embd % config.n_query_head == 0
#         assert config.n_embd % config.n_kv_head == 0
#         assert config.n_query_head % config.n_kv_head == 0
#
#         self.n_query_head = config.n_query_head
#         self.n_kv_head = config.n_kv_head
#         self.group_size = self.n_query_head // self.n_kv_head
#         self.n_embd = config.n_embd
#         self.head_dim = config.n_embd // config.n_query_head
#         self.ROPE = None
#
#         # Key, Query, Value Projections
#         self.query = nn.Linear(self.n_embd, self.n_embd)
#         self.key = nn.Linear(self.n_embd, self.head_dim * self.n_kv_head)
#         self.value = nn.Linear(self.n_embd, self.head_dim * self.n_kv_head)
#
#         # Output Projection
#         self.out = nn.Linear(self.head_dim * self.n_query_head, self.n_embd)
#
#         # regularization
#         self.attn_dropout = nn.Dropout(config.attn_pdrop)
#         self.resid_dropout = nn.Dropout(config.resid_pdrop)
#
#         self.rope = config.rope
#         if self.rope:
#             self.ROPE = RotaryPositionalEmbeddings(self.head_dim)
#
#
#     def forward(self, x, mask=None):
#         q = self.query(x)
#         k = self.key(x)
#         v = self.value(x)
#
#         B, T, C = q.size() # batch size, sequence length, embedding dimensionality (n_embd)
#
#         # split across heads by introducing an additional 'h' dimension
#         # reshape the query, key, value tensors to increase efficiency of matrix multiplication
#         # b = batch size, t = sequence length, h = number of heads, dk
#         q = rearrange(q, 'b t (h d) -> b h t d', h=self.n_query_head)
#         k = rearrange(k, 'b t (h d) -> b h t d', h=self.n_kv_head)
#         v = rearrange(v, 'b t (h d) -> b h t d', h=self.n_kv_head)
#
#         if self.rope:
#             q = self.ROPE(q)
#             k = self.ROPE(k)
#
#         q = rearrange(q, 'b (h g) t d -> b g h t d', g=self.group_size)
#
#         # compute square root of (n_embd / number of heads) to scale the dot product
#         scale = math.sqrt(k.size(-1))
#
#         # calculate the attention scores with the query and  key
#         att = einsum(q, k, 'b g h q d, b h k d -> b g h q k') / scale
#
#         if mask is not None:
#             # Unsqueeze to add dimensions for group size (G) and heads per group (H)
#             mask = mask.unsqueeze(1)  # Shape: (64, 1, 1, 86, 86)
#
#             # Expand to match attention tensor (att) shape
#             mask = mask.expand(-1, att.size(1), att.size(2), -1, -1)  # (64, 2, 4, 86, 86)
#
#             # Apply the mask
#             att = att.masked_fill(mask == 1, float('-inf'))
#
#         att_weights = F.softmax(att, dim=-1)
#         att = self.attn_dropout(att_weights)
#
#         # matrix multiplication of attention scores and value
#         y = einsum(att, v, 'b g h q k, b h k d -> b g h q d')
#
#         # rearrange the output tensor to (batch size, sequence length, n_embd)
#         y = rearrange(y, 'b g h q d -> b q (h g) d') # re-assemble all head outputs side by side
#         y = rearrange(y, 'b q h d -> b q (h d)')
#
#         # output projection
#         y = self.resid_dropout(self.out(y))
#
#         return y, att_weights

class GroupedQueryAttention:
    pass
    

class Block(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        if config.n_query_head != config.n_kv_head:
            # self.attn = GroupedQueryAttention(config)
            self.attn = None
        else:
            self.attn = MultiHeadAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = nn.ModuleDict(dict(
            c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd),
            c_proj  = nn.Linear(4 * config.n_embd, config.n_embd),
            act     = NewGELU(),
            dropout = nn.Dropout(config.resid_pdrop),
        ))
        m = self.mlp
        self.mlpf = lambda x: m.dropout(m.c_proj(m.act(m.c_fc(x)))) # MLP forward

    def forward(self, x, mask=None):
        attn_comp, attn_weights = self.attn(self.ln_1(x), mask=mask)
        x = x + attn_comp
        y = x + self.mlpf(self.ln_2(x))
        return y