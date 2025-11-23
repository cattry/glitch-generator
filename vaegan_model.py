import torch.nn as nn
import torch.nn.functional as F

# --- Constants (MUST match training values) ---
IMG_SIZE = 256
CHANNELS = 1 # Based on CHANNELS = 1 from VAE-GAN Final snippet
LATENT_DIM = 64 # Based on LATENT_DIM = 64 from VAE-GAN Final snippet

# --- VAE-GAN Encoder (Extracts features, outputs mu and logvar) ---
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1),  # 64x128x128
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.Conv2d(256, 512, 4, 2, 1),  # 512x16x16
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),

            nn.Flatten()
        )
        self.fc_mu = nn.Linear(512*16*16, LATENT_DIM)
        self.fc_logvar = nn.Linear(512*16*16, LATENT_DIM)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        # Initialize logvar to produce reasonable initial values
        nn.init.normal_(self.fc_logvar.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc_logvar.bias, 0.0)

    def forward(self, x):
        h = self.net(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

# --- VAE-GAN Generator/Decoder (Creates images from latent vector) ---
class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(LATENT_DIM, 512*16*16)

        self.net = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1),  # 64x128x128
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, CHANNELS, 4, 2, 1),  # 1x256x256
            nn.Tanh()
        )

    def forward(self, z):
        h = self.fc(z).view(-1, 512, 16, 16)
        return self.net(h)

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(CHANNELS, 64, 4, 2, 1),  # 64x128x128
            nn.LeakyReLU(0.2),

            nn.Conv2d(64, 128, 4, 2, 1),  # 128x64x64
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),

            nn.Conv2d(128, 256, 4, 2, 1),  # 256x32x32
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),

            nn.AdaptiveAvgPool2d(4),  # Smaller final feature map
            nn.Flatten(),
            nn.Linear(256*4*4, 1)
        )

    def forward(self, x):
        return self.net(x)