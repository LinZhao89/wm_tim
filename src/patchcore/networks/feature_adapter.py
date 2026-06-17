"""Feature Adaptation network for domain transfer.

This file implements a convolutional autoencoder. The encoder part can be
pre-trained on a source domain (e.g., wafermaps) and then used as a
"feature adapter" to transform input images before they are fed into a
backbone pre-trained on a different domain (e.g., ImageNet).
"""
import torch
import torch.nn as nn


class FeatureAdapter(nn.Module):
    """
    The Encoder part of the autoencoder. This is the module that will be
    saved and used for feature adaptation. It takes a 1-channel image
    and outputs a 3-channel tensor suitable for ImageNet-based backbones.
    """

    def __init__(self, in_channels=1, base_channels=32):
        super(FeatureAdapter, self).__init__()
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.bottleneck = self._conv_block(base_channels * 2, base_channels * 4)
        self.final_conv = nn.Conv2d(base_channels, 3, kernel_size=1)

    def _conv_block(self, in_c, out_c, kernel_size=3, padding=1):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, kernel_size=kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # The adapter doesn't downsample, it just transforms features.
        # This is a simplified design. A downsampling/upsampling autoencoder
        # is also a valid choice.
        x1 = self.enc1(x)
        # The output is adapted to 3 channels for the backbone.
        return torch.sigmoid(self.final_conv(x1))


class AdapterAutoencoder(nn.Module):
    """
    A full autoencoder model for pre-training the FeatureAdapter.
    It includes a decoder to reconstruct the original image.
    """

    def __init__(self, in_channels=1, base_channels=32):
        super(AdapterAutoencoder, self).__init__()
        self.encoder = FeatureAdapter(in_channels, base_channels)
        self.decoder = self._build_decoder(base_channels, in_channels)

    def _build_decoder(self, base_channels, out_channels):
        return nn.Sequential(
            nn.ConvTranspose2d(3, base_channels, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, out_channels, kernel_size=1),
            nn.Sigmoid(),  # To ensure output is in [0, 1] range like input
        )
    
    def forward(self, x):
        # A more complete autoencoder would have a symmetric decoder.
        # This is a simplified version for demonstration.
        # Let's build a more reasonable autoencoder.
        
        # Redefining the autoencoder structure to be more effective.
        self.enc1 = self.encoder.enc1
        self.enc2 = self.encoder.enc2
        self.bottleneck = self.encoder.bottleneck

        self.dec2 = self._deconv_block(32 * 4, 32 * 2)
        self.dec1 = self._deconv_block(32 * 2, 32)
        self.out_conv = nn.Conv2d(32, out_channels, kernel_size=1)

        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        b = self.bottleneck(e2)
        d2 = self.dec2(b)
        d1 = self.dec1(d2 + e2) # Skip connection
        out = self.out_conv(d1 + e1) # Skip connection
        return torch.sigmoid(out)

    def _deconv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=1, padding=0, output_padding=0),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True)
        )

    # Let's correct the autoencoder structure.
    def forward(self, x):
        # This forward pass was flawed. Let's write a proper one.
        # A simple autoencoder without downsampling:
        x1 = self.encoder.enc1(x)
        x2 = self.encoder.enc2(x1)
        b = self.encoder.bottleneck(x2)
        
        # Decoder
        d2 = self.dec2(b)
        d1 = self.dec1(d2)
        out = self.out_conv(d1)
        return torch.sigmoid(out)

# Let's try a final, cleaner autoencoder design.
class FinalAdapterAutoencoder(nn.Module):
    def __init__(self, in_channels=1, base_channels=32):
        super(FinalAdapterAutoencoder, self).__init__()
        # Encoder
        self.enc1 = self._conv_block(in_channels, base_channels)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=1)

    def _conv_block(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @property
    def adapter(self):
        # The adapter is the first encoder block + a conv to get 3 channels
        adapter_net = nn.Sequential(
            self.enc1,
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid()
        )
        return adapter_net

    def forward(self, x):
        # Encoder path
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)

        # Decoder path
        u1 = self.up1(e2)
        # Skip connection
        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        
        output = self.out_conv(d1)
        return torch.sigmoid(output)

# The above is getting complicated. Let's simplify to the core idea.
# The adapter learns to map 1-channel to 3-channels.
# The autoencoder trains this mapping by forcing reconstruction.

class Adapter(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, 3, kernel_size=1),
            nn.Sigmoid() # To make it "image-like" for the backbone
        )
    def forward(self, x):
        # Defensive: if input arrives as 1-channel, replicate to 3 channels
        if x.dim() == 4 and x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        return self.adapter(x)

class AdapterAutoencoder(nn.Module):
    def __init__(self, in_channels=3, base_channels=32):
        super(AdapterAutoencoder, self).__init__()
        self.encoder = Adapter(in_channels, base_channels)
        self.decoder = nn.Sequential(
            nn.Conv2d(3, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
