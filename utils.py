import torch
import math
from einops import rearrange


def linear_warmup_cosine_decay(warmup_steps, total_steps):
    """
    Linear warmup for warmup_steps, with cosine annealing to 0 at total_steps
    """

    def fn(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))

        progress = float(step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


def ssl_quantize(imgs, patch_size, bits_per_channel=3):
    """
    Outputs a sequence of classes, in range [0, 1, ... , 2 ** (3 * bits_per_channel) - 1]
    The sequence is formed by taking patches of size patch_size and flattening
    """
    b, *_ = imgs.shape

    # combine batch and channels
    # so convolution is done separately
    imgs = rearrange(imgs, "b c h w -> (b c) h w")

    # use convolution to find mean of each patch
    kernel = torch.ones(1, 1, patch_size, patch_size, device=imgs.device) / (patch_size ** 2)
    out = torch.nn.functional.conv2d(
        imgs.unsqueeze(1), kernel, stride=patch_size
    ).squeeze(1)

    # separate batch and channels, convert image to sequence
    out = rearrange(out, "(b c) h w -> b c (h w)", b=b)

    # reduce image from 8 bits to bits_per_channel bits
    out = out // (2 ** (8 - bits_per_channel))

    out = (
        out[:, 0]
        + (2 ** bits_per_channel) * out[:, 1]
        + (2 ** (bits_per_channel * 2)) * out[:, 2]
    )
    return out.long()