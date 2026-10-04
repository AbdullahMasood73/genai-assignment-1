"""Small models intended for constrained GPU budgets; no pretrained weights."""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def subpixel_conv(input_channels, output_channels):
    """Phase-matched initialization (ICNR) before 2x PixelShuffle."""
    layer = nn.Conv2d(input_channels, output_channels * 4, 3, 1, 1)
    with torch.no_grad():
        layer.weight.copy_(layer.weight[::4].clone().repeat_interleave(4, dim=0))
        layer.bias.zero_()
    return layer


class ResidualBlock(nn.Module):
    """Local feature refinement; no encoder-to-decoder bypass."""
    def __init__(self, channels):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(channels, channels, 3, 1, 1),
                                    nn.LeakyReLU(0.1), nn.Conv2d(channels, channels, 3, 1, 1))

    def forward(self, feature):
        return F.leaky_relu(feature + self.layers(feature) * 0.1, 0.1)


class Autoencoder(nn.Module):
    def __init__(self, base=16, latent=128, dropout=0.0, normalized=False, spatial=False, detail=False, detail_bn=False, detail_shuffle=False, **_):
        super().__init__()
        self.detail = detail
        if detail:
            if latent % 256 or not 256 <= latent < 49152:
                raise ValueError("Detail bottleneck must contain fewer than 49152 values and be divisible by 256")
            layers, previous = [], 3
            for width in (base, base * 2, base * 4):
                layers += [nn.Conv2d(previous, width, 4, 2, 1)]
                if detail_bn:
                    layers += [nn.BatchNorm2d(width)]
                layers += [nn.LeakyReLU(0.1),
                           ResidualBlock(width), nn.Dropout2d(dropout)]
                previous = width
            self.encoder = nn.Sequential(*layers)
            self.compress = nn.Conv2d(previous, latent // 256, 1)
            self.expand = nn.Conv2d(latent // 256, previous, 1)
            layers = [nn.LeakyReLU(0.1), ResidualBlock(previous)]
            for width in (base * 2, base, 3):
                layers += ([subpixel_conv(previous, width), nn.PixelShuffle(2)]
                           if detail_shuffle else
                           [nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                            nn.Conv2d(previous, width, 3, 1, 1)])
                if width == 3:
                    layers += [nn.Sigmoid()]
                else:
                    if detail_bn:
                        layers += [nn.BatchNorm2d(width)]
                    layers += [nn.LeakyReLU(0.1), ResidualBlock(width)]
                previous = width
            self.decoder = nn.Sequential(*layers)
            return
        layers = []
        previous = 3
        for width in (base, base * 2, base * 4, base * 8):
            layers += [nn.Conv2d(previous, width, 4, 2, 1)]
            if normalized:
                layers += [nn.GroupNorm(4, width), nn.LeakyReLU(0.1)]
            else:
                layers += [nn.ReLU()]
            layers += [nn.Dropout2d(dropout)]
            previous = width
        self.encoder = nn.Sequential(*layers)
        self.spatial = spatial
        if spatial:
            if latent % 64 or latent < 64:
                raise ValueError("Spatial bottleneck dimensions must be positive multiples of 64")
            self.compress = nn.Sequential(nn.Conv2d(base * 8, latent // 64, 1), nn.LeakyReLU(0.1))
            self.expand = nn.Conv2d(latent // 64, base * 8, 1)
        else:
            self.compress = nn.Linear(base * 8 * 8 * 8, latent)
            self.expand = nn.Linear(latent, base * 8 * 8 * 8)
        self.base = base
        layers = []
        for width in (base * 4, base * 2, base, 3):
            layers += ([nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        nn.Conv2d(previous, width, 3, 1, 1)] if spatial else
                       [nn.ConvTranspose2d(previous, width, 4, 2, 1)])
            if width == 3:
                layers += [nn.Sigmoid()]
            elif normalized:
                layers += [nn.GroupNorm(4, width), nn.LeakyReLU(0.1)]
            else:
                layers += [nn.ReLU()]
            previous = width
        self.decoder = nn.Sequential(*layers)

    def forward(self, image):
        encoded = self.encoder(image)
        if self.detail:
            return self.decoder(self.expand(self.compress(encoded)))
        latent = self.compress(encoded if self.spatial else encoded.flatten(1))
        feature = self.expand(latent)
        if not self.spatial:
            feature = feature.reshape(-1, self.base * 8, 8, 8)
        return self.decoder(feature)


class Classifier(nn.Module):
    def __init__(self, base=16, dropout=0.1, **_):
        super().__init__()
        layers = []
        previous = 3
        for width in (base, base * 2, base * 4, base * 8):
            layers += [nn.Conv2d(previous, width, 3, 2, 1), nn.ReLU()]
            previous = width
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(dropout), nn.Linear(previous, 4))

    def forward(self, image):
        return self.head(self.features(image))


class SoftMixture(nn.Module):
    def __init__(self, gate, experts, temperature=1.0):
        super().__init__()
        self.gate = gate
        self.experts = nn.ModuleList(experts)
        self.temperature = float(temperature)

    def forward(self, image):
        logits = self.gate(image)
        weights = torch.softmax(logits / self.temperature, dim=1)
        branches = torch.stack([image] + [expert(image) for expert in self.experts], dim=1)
        output = (branches * weights[:, :, None, None, None]).sum(1)
        return output, weights, logits


class Down(nn.Module):
    def __init__(self, input_channels, output_channels, normalized=False):
        super().__init__()
        # Refined models use tracked statistics; inference also supports batch size 1.
        layers = [nn.Conv2d(input_channels, output_channels, 4, 2, 1)]
        if normalized:
            layers += [nn.BatchNorm2d(output_channels)]
        self.block = nn.Sequential(*layers, nn.LeakyReLU(0.2))

    def forward(self, image):
        return self.block(image)


class Up(nn.Module):
    def __init__(self, input_channels, output_channels, dropout=0, refined=False):
        super().__init__()
        layers = ([subpixel_conv(input_channels, output_channels), nn.PixelShuffle(2),
                   nn.BatchNorm2d(output_channels)] if refined else
                  [nn.ConvTranspose2d(input_channels, output_channels, 4, 2, 1)])
        self.block = nn.Sequential(*layers, nn.ReLU(), nn.Dropout(dropout))

    def forward(self, image):
        return self.block(image)


class Generator(nn.Module):
    def __init__(self, base=16, embedding=8, dropout=0.1, gan_refined=False, **_):
        super().__init__()
        self.style = nn.Embedding(3, embedding)
        channels = [base, base * 2, base * 4, base * 8, base * 8]
        self.down = nn.ModuleList()
        previous = 3 + embedding
        for width in channels:
            self.down.append(Down(previous, width, normalized=gan_refined))
            previous = width
        self.up = nn.ModuleList([Up(base * 8, base * 8, dropout, gan_refined),
                                 Up(base * 16, base * 4, dropout, gan_refined),
                                 Up(base * 8, base * 2, refined=gan_refined),
                                 Up(base * 4, base, refined=gan_refined)])
        self.output = (nn.Sequential(subpixel_conv(base * 2, 3), nn.PixelShuffle(2), nn.Sigmoid())
                       if gan_refined else nn.Sequential(nn.ConvTranspose2d(base * 2, 3, 4, 2, 1), nn.Sigmoid()))

    def forward(self, image, style):
        condition = self.style(style).unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, image.shape[2], image.shape[3])
        feature = torch.cat([image, condition], dim=1)
        skips = []
        for layer in self.down:
            feature = layer(feature)
            skips.append(feature)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            feature = torch.cat([layer(feature), skip], dim=1)
        return self.output(feature)


class Discriminator(nn.Module):
    def __init__(self, base=16, embedding=8, gan_refined=False, **_):
        super().__init__()
        self.style = nn.Embedding(3, embedding)
        self.network = nn.Sequential(Down(6 + embedding, base), Down(base, base * 2, gan_refined),
                                     Down(base * 2, base * 4, gan_refined),
                                     nn.Conv2d(base * 4, 1, 3, 1, 1))

    def forward(self, photo, sketch, style):
        condition = self.style(style).unsqueeze(-1).unsqueeze(-1)
        condition = condition.expand(-1, -1, photo.shape[2], photo.shape[3])
        return self.network(torch.cat([photo, sketch, condition], dim=1))


def ssim(prediction, target):
    """11x11 Gaussian-window SSIM, data range 1, valid convolution.

    Uses the formulation in Wang et al. (2004), averaged across channels and
    spatial windows. Returns a score per image; float32 avoids AMP instability.
    """
    prediction, target = prediction.float(), target.float()
    x = torch.arange(11, device=prediction.device, dtype=torch.float32) - 5
    kernel = torch.exp(-(x * x) / (2 * 1.5 ** 2))
    kernel /= kernel.sum()
    window = (kernel[:, None] * kernel[None, :]).expand(3, 1, 11, 11).contiguous()
    def mean(tensor):
        return F.conv2d(tensor, window, groups=3)
    mu_x, mu_y = mean(prediction), mean(target)
    var_x = mean(prediction * prediction) - mu_x.square()
    var_y = mean(target * target) - mu_y.square()
    covariance = mean(prediction * target) - mu_x * mu_y
    score = ((2 * mu_x * mu_y + 0.01 ** 2) * (2 * covariance + 0.03 ** 2)) / (
        (mu_x.square() + mu_y.square() + 0.01 ** 2) * (var_x + var_y + 0.03 ** 2))
    return score.mean((1, 2, 3))


def reconstruction(output, target, alpha=0.8):
    return alpha * F.l1_loss(output.float(), target.float()) + (1 - alpha) * (1 - ssim(output, target).mean())


def build(task, cfg, components=None):
    if task in ("universal", "salt", "blur", "occlusion"):
        return Autoencoder(**cfg)
    if task == "classifier":
        return Classifier(**cfg)
    if task == "gan":
        return Generator(**cfg)
    if task == "soft":
        if components is None:
            raise ValueError("Soft mixture requires pretrained components")
        return SoftMixture(components[0], components[1:], cfg.get("temperature", 1))
    raise ValueError(task)
