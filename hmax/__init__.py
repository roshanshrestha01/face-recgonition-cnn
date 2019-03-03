# encoding: utf8
import torch.nn.functional as F
import numpy as np
from scipy.io import loadmat
import torch
from torch import nn


def gabor_filter(size, wavelength, sigma, orientation):
    lambda_ = size * 2. / wavelength
    # sigma = lambda_ * 0.8
    gamma = 0.3  # spatial aspect ratio: 0.23 < gamma < 0.92
    theta = np.deg2rad(orientation + 90)

    # Generate Gabor filter
    x, y = np.mgrid[:size, :size] - (size // 2)
    rotx = x * np.cos(theta) + y * np.sin(theta)
    roty = -x * np.sin(theta) + y * np.cos(theta)
    filt = np.exp(-(rotx**2 + gamma**2 * roty**2) / (2 * sigma ** 2))
    filt *= np.cos(2 * np.pi * rotx / lambda_)
    filt[np.sqrt(x**2 + y**2) > (size / 2)] = 0

    # Normalize the filter
    filt = filt - np.mean(filt)
    filt = filt / np.sqrt(np.sum(filt ** 2))

    return filt


class S1(nn.Module):
    def __init__(self, size, wavelength, sigma, orientations=[90, -45, 0, 45]):
        super().__init__()
        self.num_orientations = len(orientations)
        self.size = size

        self.gabor = nn.Conv2d(1, self.num_orientations, size,
                               padding=size // 2, bias=False)

        for channel, orientation in enumerate(orientations):
            self.gabor.weight.data[channel, 0] = torch.Tensor(
                gabor_filter(size, wavelength, sigma, orientation))

        self.uniform = nn.Conv2d(1, 4, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, img):
        """Apply Gabor filters, take absolute value, and normalize."""
        s1_output = torch.abs(self.gabor(img))
        norm = torch.sqrt(self.uniform(img ** 2))
        norm.data[norm == 0] = 1  # To avoid divide by zero
        s1_output /= norm
        return s1_output


class C1(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.size = size
        self.local_pool = nn.MaxPool2d(size, stride=size // 2,
                                       padding=size // 2)

    def forward(self, s1_outputs):
        """Max over scales, followed by a MaxPool2d operation."""
        s1_outputs = torch.cat([out.unsqueeze(0) for out in s1_outputs], 0)

        # Pool over all scales
        s1_output, _ = torch.max(s1_outputs, dim=0)

        # Pool over local (c1_space x c1_space) neighbourhood
        return self.local_pool(s1_output)


class S2(nn.Module):
    def __init__(self, patches, activation='gaussian', sigma=1):
        super().__init__()
        self.activation = activation
        self.sigma = sigma

        num_patches, num_orientations, size, _ = patches.shape

        # Main convolution layer
        self.conv = nn.Conv2d(in_channels=num_orientations,
                              out_channels=num_orientations * num_patches,
                              kernel_size=size,
                              padding=size // 2,
                              groups=num_orientations,
                              bias=False)
        self.conv.weight.data = torch.Tensor(
            patches.transpose(1, 0, 2, 3).reshape(1600, 1, size, size))

        # A convolution layer filled with ones. This is used for the distance
        # computation
        self.uniform = nn.Conv2d(1, 1, size, padding=size // 2, bias=False)
        nn.init.constant_(self.uniform.weight, 1)

        # This is also used for the distance computation
        self.patches_sum_sq = nn.Parameter(
            torch.Tensor((patches ** 2).sum(axis=(1, 2, 3))))

        self.num_patches = num_patches
        self.num_orientations = num_orientations
        self.size = size

        # No gradient required for this layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, c1_outputs):
        s2_outputs = []
        for c1_output in c1_outputs:
            conv_output = self.conv(c1_output)

            # Unstack the orientations
            conv_output_size = conv_output.shape[3]
            conv_output = conv_output.view(
                -1, self.num_orientations, self.num_patches, conv_output_size,
                conv_output_size)

            # Pool over orientations
            conv_output = conv_output.sum(dim=1)

            # Compute distance
            c1_sq = self.uniform(
                torch.sum(c1_output ** 2, dim=1, keepdim=True))
            dist = c1_sq - 2 * conv_output
            dist += self.patches_sum_sq[None, :, None, None]

            # Apply activation function
            if self.activation == 'gaussian':
                dist = torch.exp(- 1 / (2 * self.sigma ** 2) * dist)
            elif self.activation == 'euclidean':
                dist[dist < 0] = 0  # Negative values should never occur
                torch.sqrt_(dist)
                dist = -dist
            else:
                raise ValueError("activation parameter should be either "
                                 "'gaussian' or 'euclidean'.")

            s2_outputs.append(dist)
        return s2_outputs


class C2(nn.Module):
    """A layer of C2 units operating on a layer of S2 units."""
    def forward(self, s2_outputs):
        """Take the maximum value of the underlying S2 units."""
        maxs = [s2.max(dim=3)[0] for s2 in s2_outputs]
        maxs = [m.max(dim=2)[0] for m in maxs]
        maxs = torch.cat([m[:, None, :] for m in maxs], 1)
        return maxs.max(dim=1)[0]


class HMAX(nn.Module):
    def __init__(self, universal_patch_set, s2_act='gaussian'):
        super().__init__()

        # S1 layers, consisting of units with increasing size
        self.s1_units = [
            S1(size=7, wavelength=3.5, sigma=2.8),
            S1(size=9, wavelength=4.6, sigma=3.6),
            S1(size=11, wavelength=5.6, sigma=4.5),
            S1(size=13, wavelength=6.8, sigma=5.4),
            S1(size=15, wavelength=7.9, sigma=6.3),
            S1(size=17, wavelength=9.1, sigma=7.3),
            S1(size=19, wavelength=10.3, sigma=8.2),
            S1(size=21, wavelength=11.5, sigma=9.2),
            S1(size=23, wavelength=12.7, sigma=10.2),
            S1(size=25, wavelength=14.1, sigma=11.3),
            S1(size=27, wavelength=15.4, sigma=12.3),
            S1(size=29, wavelength=16.8, sigma=13.4),
            S1(size=31, wavelength=18.8, sigma=14.6),
            S1(size=33, wavelength=19.7, sigma=15.8),
            S1(size=35, wavelength=21.2, sigma=17.0),
            S1(size=37, wavelength=22.8, sigma=18.2),
        ]

        # Explicitly add the S1 units as submodules of the model
        for s1 in self.s1_units:
            self.add_module('s1_%02d' % s1.size, s1)

        # Each C1 layer pools across two S1 layers
        self.c1_units = [
            C1(size=8),
            C1(size=10),
            C1(size=12),
            C1(size=14),
            C1(size=16),
            C1(size=18),
            C1(size=20),
            C1(size=22),
        ]

        # Explicitly add the C1 units as submodules of the model
        for c1 in self.c1_units:
            self.add_module('c1_%02d' % c1.size, c1)

        # Read the universal patch set for the S2 layer
        m = loadmat(universal_patch_set)
        patches = [patch.reshape(shape[[2, 1, 0, 3]]).transpose(3, 0, 2, 1)
                   for patch, shape in zip(m['patches'][0], m['patchSizes'].T)]

        # One S2 layer for each patch scale, operating on all C1 layers
        self.s2_units = [S2(patches=scale_patches, activation=s2_act)
                         for scale_patches in patches]

        # Explicitly add the S2 units as submodules of the model
        for i, s2 in enumerate(self.s2_units):
            self.add_module('s2_%d' % i, s2)

        # One C2 layer operating on each scale
        self.c2_units = [C2() for s2 in self.s2_units]

        # Explicitly add the C2 units as submodules of the model
        for i, c2 in enumerate(self.c2_units):
            self.add_module('c2_%d' % i, c2)

    def run_all_layers(self, img):
        s1_outputs = [s1(img) for s1 in self.s1_units]

        # Each C1 layer pools across two S1 layers
        c1_outputs = []
        for c1, i in zip(self.c1_units, range(0, len(self.s1_units), 2)):
            c1_outputs.append(c1(s1_outputs[i:i+2]))

        s2_outputs = [s2(c1_outputs) for s2 in self.s2_units]
        c2_outputs = [c2(s2) for c2, s2 in zip(self.c2_units, s2_outputs)]

        return s1_outputs, c1_outputs, s2_outputs, c2_outputs

    def forward(self, img):
        """Run through everything and concatenate the output of the C2s."""
        c2_outputs = self.run_all_layers(img)[-1]
        c2_outputs = torch.cat(
            [c2_out[:, None, :] for c2_out in c2_outputs], 1)

        return c2_outputs

    def get_all_layers(self, img):
        s1_out, c1_out, s2_out, c2_out = self.run_all_layers(img)
        return (
            [s1.cpu().detach().numpy() for s1 in s1_out],
            [c1.cpu().detach().numpy() for c1 in c1_out],
            [[s2_.cpu().detach().numpy() for s2_ in s2] for s2 in s2_out],
            [c2.cpu().detach().numpy() for c2 in c2_out],
        )
