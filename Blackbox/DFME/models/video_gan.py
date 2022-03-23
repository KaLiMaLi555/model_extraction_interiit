# Import the required libraries
import torch.nn as nn
import torch.nn.functional as F


# VideoGAN with background process removed
class VideoGAN(nn.Module):
    def __init__(self, zdim=100):
        super(VideoGAN, self).__init__()

        self.zdim = zdim

        # Foreground
        self.conv1 = nn.ConvTranspose3d(zdim, 512, [1, 7, 7], [1, 1, 1])
        self.bn1 = nn.BatchNorm3d(512)

        self.conv2 = nn.ConvTranspose3d(512, 256, [1, 4, 4], [1, 2, 2], [0, 1, 1])
        self.bn2 = nn.BatchNorm3d(256)

        self.conv3 = nn.ConvTranspose3d(256, 128, [4, 4, 4], [2, 2, 2], [1, 1, 1])
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.ConvTranspose3d(128, 64, [4, 4, 4], [2, 2, 2], [1, 1, 1])
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.ConvTranspose3d(64, 32, [4, 4, 4], [2, 2, 2], [1, 1, 1])
        self.bn5 = nn.BatchNorm3d(32)

        self.conv6 = nn.ConvTranspose3d(32, 3, [4, 4, 4], [2, 2, 2], [1, 1, 1])

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z, pre_x):
        # Foreground
        z_unsqueezed = z.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        f = F.leaky_relu(self.bn1(self.conv1(z_unsqueezed)))
        f = F.leaky_relu(self.bn2(self.conv2(f)))
        f = F.leaky_relu(self.bn3(self.conv3(f)))
        f = F.leaky_relu(self.bn4(self.conv4(f)))
        f = F.leaky_relu(self.bn5(self.conv5(f)))

        out = self.conv6(f)  # b, 3, 32, 64, 64
        if pre_x:
            return out
        else:
            return torch.sigmoid(out)
