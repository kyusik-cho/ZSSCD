import torch
import torch.nn as nn
from einops import rearrange
import math

class AdaptiveInstanceNormalization(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNormalization, self).__init__()
        self.init_style()
        self.need_style_update = True

    def forward(self, feat):
        if self.need_style_update: # update the style, and adain does not work for in-domain frame
            self.update_style(feat)
            restylized_feat = feat

        else:
            size = feat.size()
            content_mean, content_std = calc_mean_std(feat)

            normalized_feat = (feat - content_mean.expand(size))/content_std.expand(size)
            restylized_feat = normalized_feat * self.style_std.expand(size) + self.style_mean.expand(size)

        return restylized_feat

    def init_style(self):
        self.style_mean, self.style_std = None, None

    def update_style(self, x_style):
        self.style_mean, self.style_std = calc_mean_std(x_style)

class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

def calc_mean_std(feat, eps=1e-5):
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_std, feat_mean = torch.std_mean(feat.view(N, C, -1), dim=2)
    return feat_mean.view(N, C, 1, 1), feat_std.view(N, C, 1, 1) + eps
    