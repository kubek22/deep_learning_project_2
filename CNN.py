import torch
import torch.nn as nn
import torch.nn.functional as F

class AudioClassifier(nn.Module):
    def __init__(self, num_classes=11, drop=0.3):
        super(AudioClassifier, self).__init__()
        
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=512, kernel_size=240, stride=15)
        self.bn1 = nn.BatchNorm1d(512)
        self.drop1 = nn.Dropout(drop)
        
        self.conv2 = nn.Conv1d(in_channels=512, out_channels=512, kernel_size=5, stride=1)
        self.bn2 = nn.BatchNorm1d(512)
        self.drop2 = nn.Dropout(drop)
        
        self.conv3 = nn.Conv1d(in_channels=512, out_channels=256, kernel_size=3, stride=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.drop3 = nn.Dropout(drop)

        self.bilstm1 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.drop4 = nn.Dropout(drop)
        self.bn4 = nn.BatchNorm1d(256)

        self.bilstm2 = nn.LSTM(input_size=256, hidden_size=128, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.drop5 = nn.Dropout(drop)
        self.bn5 = nn.BatchNorm1d(256)

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc1 = nn.Linear(256, 256)
        self.drop6 = nn.Dropout(drop)
        self.fc2 = nn.Linear(256, 256)
        self.drop7 = nn.Dropout(drop)
        self.fc3 = nn.Linear(256, 128)
        self.drop8 = nn.Dropout(drop)

        self.fc_out = nn.Linear(128, num_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)

        x = self.drop1(self.bn1(self.conv1(x)))    
        x = self.drop2(self.bn2(self.conv2(x)))   
        x = self.drop3(self.bn3(self.conv3(x))) 

        x = x.permute(0, 2, 1)
        x, _ = self.bilstm1(x)
        x = self.drop4(x)
        x = self.bn4(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)

        x, _ = self.bilstm2(x)
        x = self.drop5(x)
        x = self.bn5(x.permute(0, 2, 1))

        x = self.global_pool(x)
        x = x.squeeze(2)

        x = self.drop6(F.relu(self.fc1(x)))
        x = self.drop7(F.relu(self.fc2(x)))
        x = self.drop8(F.relu(self.fc3(x)))

        out = self.fc_out(x)
        return out