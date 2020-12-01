import torch
from torch import nn
from einops import rearrange


class MultiheadAttention(nn.Module):
    """
    Multihead Attention
    Forwadr expecte shape [batch, sequence len, dim]
    """
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.wq = nn.Linear(dim, dim)
        self.wk = nn.Linear(dim, dim)
        self.wv = nn.Linear(dim, dim)
        self.wout = nn.Linear(dim, dim)

    def forward(self, x):
        q = rearrange(self.wq(x), "b l (head d) -> b head l d", head=self.heads)
        k = rearrange(self.wk(x), "b l (head d) -> b head l d", head=self.heads)
        v = rearrange(self.wv(x), "b l (head d) -> b head l d", head=self.heads)

        attn = torch.einsum("bhid,bhjd->bhij", q, k) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, "b head l d -> b l (head d)")
        return self.wout(out)


class Block(nn.Module):
    def __init__(self, dim, heads, dropout_p=0.0):
        super().__init__()
        self.ln_1 = nn.LayerNorm(dim)
        self.ln_2 = nn.LayerNorm(dim)
        self.attn = MultiheadAttention(dim, heads)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim)
        )
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        x = self.ln_1(x)
        a = self.attn(x)
        x = x + self.dropout(a)
        m = self.mlp(self.ln_2(x))
        x = x + self.dropout(m)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, heads, layers, dropout_p=0.0):
        super().__init__()
        self.layers = nn.ModuleList(
            [Block(dim, heads, dropout_p) for _ in range(layers)]
        )
        self.ln_f = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Expects inputs of shape [batch, sequence length, dim]
        """
        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)
        return x


def test_transformer():
    performer = Transformer(dim=8, heads=2, layers=2)
    x = torch.rand(64, 100, 8) # batch, sequence length, dim
    out = performer(x)
    assert out.shape == (64, 100, 8)
