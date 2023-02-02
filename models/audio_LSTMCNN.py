from typing import Tuple
import torch
from torch import nn


class AudioLSTMCNN(nn.Module):
    def __init__(self, out_size: int = 2, cnn_channels: int = 64):
        """
        For a spectrograms with 128 buckets and chunk size of 196, will be (128, 196)
        """
        # call the parent constructor
        super(AudioLSTMCNN, self).__init__()

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1),
                                padding=1)
        self.relu12 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv21 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1),
                                padding=1)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1),
                                padding=1)
        self.relu22 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout4 = nn.Dropout(p=0.25)

        self.conv5 = nn.Conv2d(in_channels=cnn_channels * 4, out_channels=cnn_channels * 4, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout5 = nn.Dropout(p=0.25)

        self.lstm6 = nn.LSTM(cnn_channels * 4, cnn_channels * 4, batch_first=True)
        self.hidden = (torch.zeros(1, 1, cnn_channels * 4),
                       torch.zeros(1, 1, cnn_channels * 4))
        self.fc6 = nn.Linear(in_features=cnn_channels * 4, out_features=cnn_channels * 4)
        self.dropout6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(in_features=cnn_channels * 4, out_features=cnn_channels * 4)
        self.dropout7 = nn.Dropout(p=0.5)

        self.fc8 = nn.Linear(in_features=cnn_channels * 4, out_features=out_size)
        self.final = nn.Identity()

    def forward(self, x):
        x = x.reshape((1, 1, x.shape[0], -1))

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.maxpool1(x)

        x = self.conv21(x)
        x = self.relu21(x)
        x = self.conv22(x)
        x = self.relu22(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = self.dropout5(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)

        x, self.hidden = self.lstm6(x, self.hidden)
        x = self.fc6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.dropout7(x)

        x = self.fc8(x)

        final_x = self.final(x.reshape((-1)))

        return final_x


class AudioLSTMCNN2(nn.Module):
    def __init__(self, out_size: int = 2, cnn_channels: int = 64):
        """
        For a spectrograms with 128 buckets and chunk size of 196, will be (128, 196)
        """
        # call the parent constructor
        super(AudioLSTMCNN2, self).__init__()

        self.conv11 = nn.Conv2d(in_channels=1, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.relu11 = nn.ReLU()
        self.conv12 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1),
                                padding=1)
        self.relu12 = nn.ReLU()
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv21 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1),
                                padding=1)
        self.relu21 = nn.ReLU()
        self.conv22 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels, kernel_size=(3, 3), stride=(1, 1),
                                padding=1)
        self.relu22 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))

        self.conv3 = nn.Conv2d(in_channels=cnn_channels, out_channels=cnn_channels * 2, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.relu3 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout3 = nn.Dropout(p=0.25)

        self.conv4 = nn.Conv2d(in_channels=cnn_channels * 2, out_channels=cnn_channels * 4, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.relu4 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout4 = nn.Dropout(p=0.25)

        self.conv5 = nn.Conv2d(in_channels=cnn_channels * 4, out_channels=cnn_channels * 4, kernel_size=(3, 3),
                               stride=(1, 1), padding=1)
        self.relu5 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=(3, 3), stride=(3, 3))
        self.dropout5 = nn.Dropout(p=0.25)

        self.lstm6 = nn.LSTM(cnn_channels * 4, cnn_channels * 4)  # , batch_first=True)
        self.hidden = (torch.zeros(1, 1, cnn_channels * 4),
                       torch.zeros(1, 1, cnn_channels * 4))
        self.fc6 = nn.Linear(in_features=cnn_channels * 4, out_features=cnn_channels * 2)
        self.dropout6 = nn.Dropout(p=0.5)

        self.fc7 = nn.Linear(in_features=cnn_channels * 2, out_features=cnn_channels)
        self.dropout7 = nn.Dropout(p=0.5)

        self.fc8 = nn.Linear(in_features=cnn_channels, out_features=cnn_channels//2)
        self.fc9 = nn.Linear(in_features=cnn_channels//2, out_features=cnn_channels//4)
        self.fc10 = nn.Linear(in_features=cnn_channels//4, out_features=out_size)
        self.final = nn.Identity()

    def forward(self, x):
        x = x.reshape((1, 1, x.shape[0], -1))

        x = self.conv11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.relu12(x)
        x = self.maxpool1(x)

        x = self.conv21(x)
        x = self.relu21(x)
        x = self.conv22(x)
        x = self.relu22(x)
        x = self.maxpool2(x)

        x = self.conv3(x)
        x = self.relu3(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)

        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)

        x = self.conv5(x)
        x = self.relu5(x)
        x = self.maxpool5(x)
        x = self.dropout5(x)

        x = x.view(x.size(0), x.size(1), -1)
        x = x.permute(0, 2, 1)

        x, self.hidden = self.lstm6(x, self.hidden)

        x = x.view(x.size(0), -1)
        x = self.fc6(x)
        x = self.dropout6(x)

        x = self.fc7(x)
        x = self.dropout7(x)

        x = self.fc8(x)
        x = self.fc9(x)
        x = self.fc10(x)

        final_x = self.final(x.reshape((-1)))

        return final_x
