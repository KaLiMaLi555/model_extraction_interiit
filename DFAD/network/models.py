# Import the required libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class DecoderRNN(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, drop_p=0.3, num_classes=400):
        super(DecoderRNN, self).__init__()

        self.RNN_input_size = CNN_embed_dim
        self.h_RNN_layers = h_RNN_layers  # RNN hidden layers
        self.h_RNN = h_RNN  # RNN hidden nodes
        self.h_FC_dim = h_FC_dim
        self.drop_p = drop_p
        self.num_classes = num_classes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            hidden_size=self.h_RNN,
            num_layers=h_RNN_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)
        """ h_n shape (n_layers, batch, hidden_size), h_c shape (n_layers, batch, hidden_size) """
        """ None represents zero initial hidden state. RNN_out has shape=(batch, time_step, output_size) """

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


# 2D CNN encoder using ResNet-152 pretrained
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, pretrained=False):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        # Model Update: Unfreezing more layers when pretrained
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        # Model Update: Freezing only when pretrained
        if pretrained:
            for module in modules[:-2]:
                for param in module.parameters():
                    param.requires_grad = False
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, fc_hidden1)
        self.bn1 = nn.BatchNorm1d(fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(fc_hidden1, fc_hidden2)
        self.bn2 = nn.BatchNorm1d(fc_hidden2, momentum=0.01)
        self.fc3 = nn.Linear(fc_hidden2, CNN_embed_dim)

    def forward(self, x_3d):
        cnn_embed_seq = []
        for t in range(x_3d.size(1)):
            # ResNet CNN
            x = x_3d[:, t, :, :, :]
            # NOTE: Uncomment when the shape of input video is (b, c, f, h, w)
            # x = x.reshape((-1, x.shape[3], x.shape[1], x.shape[2]))

            x = self.resnet(x)  # ResNet
            x = x.view(x.size(0), -1)  # flatten output of conv

            # FC layers
            x = self.bn1(self.fc1(x))
            x = F.relu(x)
            x = self.bn2(self.fc2(x))
            x = F.relu(x)
            # Model update: Not using dropout with BN
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)

        return cnn_embed_seq


class ResCNNRNN(nn.Module):

    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128, num_classes=400):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(ResCNNRNN, self).__init__()

        self.encoder = ResCNNEncoder(fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2, drop_p=drop_p, CNN_embed_dim=CNN_embed_dim)
        self.decoder = DecoderRNN(CNN_embed_dim=CNN_embed_dim, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN, h_FC_dim=h_FC_dim, drop_p=drop_p, num_classes=num_classes)

    def forward(self, x_3d):
        return self.decoder(self.encoder(x_3d))


class VideoGAN(nn.Module):
    def __init__(self, zdim=100):
        super(VideoGAN, self).__init__()

        self.zdim = zdim

        # Background
        # self.conv1b = nn.ConvTranspose2d(zdim, 512, [4,4], [1,1])
        # self.bn1b = nn.BatchNorm2d(512)

        # self.conv2b = nn.ConvTranspose2d(512, 256, [4,4], [2,2], [1,1])
        # self.bn2b = nn.BatchNorm2d(256)

        # self.conv3b = nn.ConvTranspose2d(256, 128, [4,4], [2,2], [1,1])
        # self.bn3b = nn.BatchNorm2d(128)

        # self.conv4b = nn.ConvTranspose2d(128, 64, [4,4], [2,2], [1,1])
        # self.bn4b = nn.BatchNorm2d(64)

        # self.conv5b = nn.ConvTranspose2d(64, 3, [4,4], [2,2], [1,1])

        # Foreground
        self.conv1 = nn.ConvTranspose3d(zdim, 512, [1, 4, 4], [1, 1, 1])
        self.bn1 = nn.BatchNorm3d(512)

        self.conv2 = nn.ConvTranspose3d(512, 256, [4, 4, 4], [2, 2, 2], [1, 1, 1])
        self.bn2 = nn.BatchNorm3d(256)

        self.conv3 = nn.ConvTranspose3d(256, 128, [4, 4, 4], [2, 2, 2], [1, 1, 1])
        self.bn3 = nn.BatchNorm3d(128)

        self.conv4 = nn.ConvTranspose3d(128, 64, [4, 4, 4], [2, 2, 2], [1, 1, 1])
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.ConvTranspose3d(64, 3, [4, 4, 4], [2, 2, 2], [1, 1, 1])

        # Mask
        # self.conv5m = nn.ConvTranspose3d(64, 1, [4,4,4], [2,2,2], [1,1,1])

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
        # b = F.relu(self.bn1b(self.conv1b(z.unsqueeze(2).unsqueeze(3))))
        # b = F.relu(self.bn2b(self.conv2b(b)))
        # b = F.relu(self.bn3b(self.conv3b(b)))
        # b = F.relu(self.bn4b(self.conv4b(b)))
        # b = torch.tanh(self.conv5b(b)).unsqueeze(2)  # b, 3, 1, 64, 64

        # Foreground
        f = F.relu(self.bn1(self.conv1(z.unsqueeze(2).unsqueeze(3).unsqueeze(4))))
        f = F.relu(self.bn2(self.conv2(f)))
        f = F.relu(self.bn3(self.conv3(f)))
        f = F.relu(self.bn4(self.conv4(f)))
        # m = torch.sigmoid(self.conv5m(f))   # b, 1, 32, 64, 64
        f = torch.tanh(self.conv5(f))  # b, 3, 32, 64, 64

        # out = m*f + (1-m)*b
        out = f

        return out


class ImageGenerator(nn.Module):
    def __init__(self, ngpu, nc=3, nz=100, ngf=64):
        super(ImageGenerator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            nn.ConvTranspose2d(ngf, nc, kernel_size=1, stride=1, padding=0, bias=False),
        )

    def forward(self, x):
        if x.is_cuda and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, x, range(self.ngpu))
        else:
            output = self.main(x)
        return output
