# src/models.py

import torch
import torch.nn as nn

class PolicyNet(nn.Module):
    """
    CNN + MLP policy that takes:
      - RGB image (3 x 64 x 64)
      - noisy cube position (3,)
    and outputs:
      - mean and log_std for a 2D correction (dx, dy).
    """
    def __init__(self):
        super().__init__()

        # Convolutional branch for image (3 x 64 x 64)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2),  # 16 x 32 x 32
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=5, stride=2, padding=2), # 32 x 16 x 16
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2), # 64 x 8 x 8
            nn.ReLU(),
            nn.Flatten(),                                          # 64 * 8 * 8 = 4096
        )

        cnn_out_dim = 64 * 8 * 8

        # MLP branch for noisy cube position (3D)
        self.mlp_pos = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Combined head
        self.fc = nn.Sequential(
            nn.Linear(cnn_out_dim + 64, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
        )

        # Output: mean and log_std for 2D correction (x, y)
        self.mean_head = nn.Linear(64, 2)
        self.log_std_head = nn.Linear(64, 2)

    def forward(self, img, noisy_pos):
        """
        img:  (B, 3, 64, 64)
        noisy_pos: (B, 3)
        returns: mean, log_std each (B, 2)
        """
        img_feat = self.cnn(img)
        pos_feat = self.mlp_pos(noisy_pos)
        x = torch.cat([img_feat, pos_feat], dim=-1)
        x = self.fc(x)

        mean = self.mean_head(x)
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, -5, 2)  # keep std reasonable

        return mean, log_std
