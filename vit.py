
import torch
from torch import nn
from einops import rearrange

from transformer import Transformer


class VisionTransformer(nn.Module):

    def __init__(
        self,
        image_size,
        patch_size,
        num_classes,
        dim,
        layers,
        heads,
        classifier="gap",
        dropout_p=0.0,
        ssl_out=512,
    ):
        super().__init__()

        assert image_size % patch_size == 0, "you're gonna have a bad time"
        num_patches = (image_size // patch_size) ** 2

        self.dim = dim
        self.classifier = classifier

        # init positional embedding with 0.02 from BERT
        self.pos_emb = nn.Parameter(
            torch.randn(1, num_patches + (classifier == "token"), dim) * 0.02
        )
        self.patch_emb = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)

        if classifier == "token":
            self.clf_token = nn.Parameter(torch.zeros(dim))

        self.transformer = Transformer(dim, heads, layers, dropout_p)
        self.ln = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        self.ssl_head = nn.Linear(dim, ssl_out)

    def forward(self, x, ssl=False):
        batch, *_ = x.shape

        x = self.patch_emb(x)

        # flatten patches into sequence
        x = rearrange(x, "b c h w -> b (h w) c")

        if self.classifier == "token":
            # prepend classification token
            clf_token = torch.ones(1, batch, self.dim, device=x.device) * self.clf_token
            x = torch.cat([clf_token, x[:-1, :, :]], axis=0)

        # add position embedding
        x += self.pos_emb.expand_as(x)

        x = self.transformer(x)

        x = self.ln(x)

        if ssl:
            x = self.ssl_head(x) # predict for each element in sequence
            return x

        if self.classifier == "token":
            x = x[:, 0]
        elif self.classifier == "gap":
            x = x.mean(dim=1)  # mean over sequence len

        x = self.head(x)

        return x


def test_vision_transformer():
    vit = VisionTransformer(
        image_size=32, patch_size=4, num_classes=10, dim=128, layers=8, heads=8
    )
    img = torch.randn(64, 3, 32, 32)
    out = vit(img)
    assert out.shape == (64, 10)
