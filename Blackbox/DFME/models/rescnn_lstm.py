import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


# 2D CNN encoder using ResNet-50 architecture
class ResCNNEncoder(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3, CNN_embed_dim=300):
        """Load the ResNet-50 architecture and replace top fc layer."""
        super(ResCNNEncoder, self).__init__()

        self.fc_hidden1, self.fc_hidden2 = fc_hidden1, fc_hidden2
        self.drop_p = drop_p

        resnet = torchvision.models.resnet50(pretrained=False)
        modules = list(resnet.children())[:-1]  # Delete the last FC layer
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
            x = self.fc3(x)

            cnn_embed_seq.append(x)

        # Swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape (batch, time_step, input_size)
        return cnn_embed_seq


# LSTM Decoder
class LSTMDecoder(nn.Module):
    def __init__(self, CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256,
                 h_FC_dim=128, drop_p=0.3, num_classes=400):
        super(LSTMDecoder, self).__init__()

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
            # input & output will have batch size as 1st dimension.
            # Eg: (batch, time_step, input_size)
            batch_first=True,
        )

        self.fc1 = nn.Linear(self.h_RNN, self.h_FC_dim)
        self.fc2 = nn.Linear(self.h_FC_dim, self.num_classes)

    def forward(self, x_RNN):
        self.LSTM.flatten_parameters()
        # RNN_out has shape (batch, time_step, output_size),
        # h_n shape and h_c shape (n_layers, batch, hidden_size).
        # None represents zero initial hidden state.
        RNN_out, (h_n, h_c) = self.LSTM(x_RNN, None)

        # FC layers
        x = self.fc1(RNN_out[:, -1, :])  # Choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


# ResCNN-LSTM model for video classification
class ResCNNLSTM(nn.Module):
    def __init__(self, fc_hidden1=512, fc_hidden2=512, drop_p=0.3,
                 CNN_embed_dim=300, h_RNN_layers=3, h_RNN=256, h_FC_dim=128,
                 num_classes=400):
        super(ResCNNLSTM, self).__init__()

        self.encoder = ResCNNEncoder(
            fc_hidden1=fc_hidden1, fc_hidden2=fc_hidden2,
            drop_p=drop_p, CNN_embed_dim=CNN_embed_dim)

        self.decoder = LSTMDecoder(
            CNN_embed_dim=CNN_embed_dim, h_RNN_layers=h_RNN_layers, h_RNN=h_RNN,
            h_FC_dim=h_FC_dim, drop_p=drop_p, num_classes=num_classes)

    def forward(self, x_3d):
        return self.decoder(self.encoder(x_3d))
