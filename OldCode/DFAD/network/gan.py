import torch
import torch.nn as nn
import torch.nn.functional as F

class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.shape[0], -1)

class GeneratorA(nn.Module):
    def __init__(self, nz=100, ngf=64, nc=1, img_size=32):
        super(GeneratorA, self).__init__()

        self.init_size = img_size//4
        self.l1 = nn.Sequential(nn.Linear(nz, ngf*2*self.init_size**2))

        self.conv_blocks0 = nn.Sequential(
            nn.BatchNorm2d(ngf*2),
        )
        self.conv_blocks1 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf*2, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.conv_blocks2 = nn.Sequential(
            nn.Conv2d(ngf*2, ngf, 3, stride=1, padding=1),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ngf, nc, 3, stride=1, padding=1),
            nn.Tanh(),
            nn.BatchNorm2d(nc, affine=False) 
        )

    def forward(self, z):
        out = self.l1(z.view(z.shape[0],-1))
        out = out.view(out.shape[0], -1, self.init_size, self.init_size)
        img = self.conv_blocks0(out)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks1(img)
        img = nn.functional.interpolate(img,scale_factor=2)
        img = self.conv_blocks2(img)
        return img


class GeneratorB(nn.Module):
    """ Generator from DCGAN: https://arxiv.org/abs/1511.06434
    """
    def __init__(self, nz=256, ngf=64, nc=3, img_size=64, slope=0.2):
        super(GeneratorB, self).__init__()
        if isinstance(img_size, (list, tuple)):
            self.init_size = ( img_size[0]//16, img_size[1]//16 )
        else:    
            self.init_size = ( img_size // 16, img_size // 16)

        self.project = nn.Sequential(
            Flatten(),
            nn.Linear(nz, ngf*8*self.init_size[0]*self.init_size[1]),
        )

        self.main = nn.Sequential(
            nn.BatchNorm2d(ngf*8),
            
            nn.ConvTranspose2d(ngf*8, ngf*4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*4),
            nn.LeakyReLU(slope, inplace=True),
            # 2x

            nn.ConvTranspose2d(ngf*4, ngf*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf*2),
            nn.LeakyReLU(slope, inplace=True),
            # 4x
            
            nn.ConvTranspose2d(ngf*2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 8x

            nn.ConvTranspose2d(ngf, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.LeakyReLU(slope, inplace=True),
            # 16x

            nn.Conv2d(ngf, nc, 3,1,1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, (nn.ConvTranspose2d, nn.Linear, nn.Conv2d)):
                nn.init.normal_(m.weight, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, (nn.BatchNorm2d)):
                nn.init.normal_(m.weight, 1.0, 0.02)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, z):
        proj = self.project(z)
        proj = proj.view(proj.shape[0], -1, self.init_size[0], self.init_size[1])
        output = self.main(proj)
        return output



class VideoGAN(nn.Module):
    def __init__(self, zdim = 100):
        super(VideoGAN, self).__init__()
        
        self.zdim = zdim
        
        # Background
        self.conv1b = nn.ConvTranspose2d(zdim, 512, [4,4], [1,1])
        self.bn1b = nn.BatchNorm2d(512)

        self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        self.bn2b = nn.BatchNorm2d(256)

        self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [2,2], [1,1])
        self.bn3b = nn.BatchNorm2d(128)

        self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        self.bn4b = nn.BatchNorm2d(64)

        self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [2,2], [1,1])

        # For getting 224 x 224
        # self.conv1b = nn.ConvTranspose2d(zdim, 512, [4,4], [2,2])
        # self.bn1b = nn.BatchNorm2d(512)

        # self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        # self.bn2b = nn.BatchNorm2d(256)

        # self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [4,4], [1,1])
        # self.bn3b = nn.BatchNorm2d(128)

        # self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        # self.bn4b = nn.BatchNorm2d(64)

        # self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [7,7], [1,1])

        # Foreground
        self.conv1 = nn.ConvTranspose3d(zdim, 512, [1,4,4], [1,1,1])
        self.bn1 = nn.BatchNorm3d(512)

        self.conv2 = nn.ConvTranspose3d(512, 256, [4,4,4], [2,2,2], [1,1,1])
        self.bn2 = nn.BatchNorm3d(256)

        self.conv3 = nn.ConvTranspose3d(256, 128, [4,4,4], [2,2,2], [1,1,1])
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.ConvTranspose3d(128, 64, [4,4,4], [2,2,2], [1,1,1])
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.ConvTranspose3d(64, 3, [4,4,4], [2,2,2], [1,1,1])

        # For getting 224 x 224
        # self.conv1 = nn.ConvTranspose3d(zdim, 512, [1,4,4], [2,2,2])
        # self.bn1 = nn.BatchNorm3d(512)

        # self.conv2 = nn.ConvTranspose3d(512, 256, [4,4,4], [2,2,2])
        # self.bn2 = nn.BatchNorm3d(256)

        # self.conv3 = nn.ConvTranspose3d(256, 128, [4,4,4], [4,4,4])
        # self.bn3 = nn.BatchNorm3d(128)

        # self.conv4 = nn.ConvTranspose3d(128, 64, [4,4,4], [2,2,2])
        # self.bn4 = nn.BatchNorm3d(64)

        # self.conv5 = nn.ConvTranspose3d(64, 3, [4,4,4], [7,7,7])

        # self.conv1 = nn.ConvTranspose3d(zdim, 256, [1,4,4])
        # self.bn1 = nn.BatchNorm3d(256)

        # self.conv2 = nn.ConvTranspose3d(256, 128, [4,4,4], [1,1,1], [1,1,1])
        # self.bn2 = nn.BatchNorm3d(128)

        # self.conv3 = nn.ConvTranspose3d(128, 64, [4,4,4], [1,1,1], [1,1,1])
        # self.bn3 = nn.BatchNorm3d(64)

        # self.conv4 = nn.ConvTranspose3d(64, 16, [4,4,4], [2,2,2], [1,1,1])
        # self.bn4 = nn.BatchNorm3d(16)

        # self.conv5 = nn.ConvTranspose3d(16, 3, [4,4,4], [2,2,2], [1,1,1])

        # Mask
        self.conv5m = nn.ConvTranspose3d(64, 1, [4,4,4], [2,2,2], [1,1,1])

        # Init weights
        for m in self.modules():
            classname = m.__class__.__name__
            if classname.lower().find('conv') != -1:
                nn.init.normal_(m.weight, mean=0, std=0.01)
                nn.init.constant_(m.bias, 0)
            elif classname.lower().find('bn') != -1:
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, z):
        # Background
        b = F.relu(self.bn1b(self.conv1b(z.unsqueeze(2).unsqueeze(3))))
        # print(b.shape)
        b = F.relu(self.bn2b(self.conv2b(b)))
        # print(b.shape)
        b = F.relu(self.bn3b(self.conv3b(b)))
        # print(b.shape)
        b = F.relu(self.bn4b(self.conv4b(b)))
        # print(b.shape)
        b = torch.tanh(self.conv5b(b)).unsqueeze(2)  # b, 3, 1, 64, 64

        # Foreground
        f = F.relu(self.bn1(self.conv1(z.unsqueeze(2).unsqueeze(3).unsqueeze(4))))
        # print(f.shape)
        f = F.relu(self.bn2(self.conv2(f)))
        # print(f.shape)
        f = F.relu(self.bn3(self.conv3(f)))
        # print(f.shape)
        f = F.relu(self.bn4(self.conv4(f)))
        # print(f.shape)
        m = torch.sigmoid(self.conv5m(f))   # b, 1, 32, 64, 64
        f = torch.tanh(self.conv5(f))   # b, 3, 32, 64, 64
        # print(f.shape)
        
        out = m*f + (1-m)*b

        # return out, f, b, m
        return out