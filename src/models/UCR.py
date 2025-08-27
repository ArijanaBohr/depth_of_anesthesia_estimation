import torch
import torch.nn as nn

class SimpleModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleModel, self).__init__()
        self.dropout1 = nn.Dropout(0.1)
        self.dense1 = nn.Linear(input_size, hidden_size)
        self.dropout2 = nn.Dropout(0.2)
        self.dense2 = nn.Linear(hidden_size, hidden_size)
        self.dropout3 = nn.Dropout(0.2)
        self.dense3 = nn.Linear(hidden_size, hidden_size)
        self.dropout4 = nn.Dropout(0.3)
        self.out = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid() 

    def forward(self, x):
        x = x.squeeze(-1)
        x = self.dropout1(x)
        x = torch.relu(self.dense1(x))
        x = self.dropout2(x)
        x = torch.relu(self.dense2(x))
        x = self.dropout3(x)
        x = torch.relu(self.dense3(x))
        x = self.dropout4(x)
        x = self.out(x)
        x = self.sigmoid(x)
        return x
    
class FCN(nn.Module):
    def __init__(self, input_shape, num_classes, bn=True):
        super(FCN, self).__init__()
        self.conv1 = nn.Conv2d(input_shape, 128, kernel_size=(8, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=(5, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(256)
        self.conv3 = nn.Conv2d(256, 128, kernel_size=(3, 1), padding='same')
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)
        self.sigmoid = nn.Sigmoid() 
        self.bn=bn

    def forward(self, x):
        if len(x.shape)<3:
            x = x.unsqueeze(1)
        elif len(x.shape)>3:
            x = x.squeeze(1)
        else:
            x = x
        x = x.unsqueeze(-1)
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        if self.bn:
            x = self.bn3(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
    
    
class UCRResNet(nn.Module):
    def __init__(self, input_shape, n_feature_maps, nb_classes,bn=False):
        super(UCRResNet, self).__init__()
        self.bn0 = nn.BatchNorm2d(input_shape)
        self.conv1 = nn.Conv2d(input_shape, n_feature_maps, kernel_size=(8, 1), padding='same')
        self.bn1 = nn.BatchNorm2d(n_feature_maps)
        self.conv2 = nn.Conv2d(n_feature_maps, n_feature_maps, kernel_size=(5, 1), padding='same')
        self.bn2 = nn.BatchNorm2d(n_feature_maps)
        self.conv3 = nn.Conv2d(n_feature_maps, n_feature_maps, kernel_size=(3, 1), padding='same')
        self.bn3 = nn.BatchNorm2d(n_feature_maps)

        self.shortcut1 = self._make_shortcut(input_shape, n_feature_maps)

        self.conv4 = nn.Conv2d(n_feature_maps, n_feature_maps * 2, kernel_size=(8, 1), padding='same')
        self.bn4 = nn.BatchNorm2d(n_feature_maps * 2)
        self.conv5 = nn.Conv2d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=(5, 1), padding='same')
        self.bn5 = nn.BatchNorm2d(n_feature_maps * 2)
        self.conv6 = nn.Conv2d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=(3, 1), padding='same')
        self.bn6 = nn.BatchNorm2d(n_feature_maps * 2)

        self.shortcut2 = self._make_shortcut(n_feature_maps, n_feature_maps * 2)

        self.conv7 = nn.Conv2d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=(8, 1), padding='same')
        self.bn7 = nn.BatchNorm2d(n_feature_maps * 2)
        self.conv8 = nn.Conv2d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=(5, 1), padding='same')
        self.bn8 = nn.BatchNorm2d(n_feature_maps * 2)
        self.conv9 = nn.Conv2d(n_feature_maps * 2, n_feature_maps * 2, kernel_size=(3, 1), padding='same')
        self.bn9 = nn.BatchNorm2d(n_feature_maps * 2)

        self.shortcut3 = self._make_shortcut(n_feature_maps * 2, n_feature_maps * 2)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(n_feature_maps * 2, nb_classes)
        self.sigmoid = nn.Sigmoid()
        self.bn =bn

    def _make_shortcut(self, in_channels, out_channels):
        if in_channels != out_channels:
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding='same'),
                nn.BatchNorm2d(out_channels)
            )
        else:
            return nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if len(x.shape)<3:
            x = x.unsqueeze(1)
        elif len(x.shape)>3:
            x = x.squeeze(1)
        else:
            x = x
        x = x.unsqueeze(-1)
        if self.bn:
            x = self.bn0(x)

        shortcut = self.shortcut1(x)
        x = self.conv1(x)
        if self.bn:
            x = self.bn1(x)
        x = torch.relu(x)
        x = self.conv2(x)
        if self.bn:
            x = self.bn2(x)
        x = torch.relu(x)
        x = self.conv3(x)
        if self.bn:
            x = self.bn3(x)
        x = torch.relu(x + shortcut)

        shortcut = self.shortcut2(x)
        x = self.conv4(x)
        if self.bn:
            x = self.bn4(x)
        x = torch.relu(x)
        x = self.conv5(x)
        if self.bn:
            x = self.bn5(x)
        x = torch.relu(x)
        x = self.conv6(x)
        if self.bn:
            x = self.bn6(x)
        x = torch.relu(x + shortcut)

        shortcut = self.shortcut3(x)
        x = self.conv7(x)
        if self.bn:
            x = self.bn7(x)
        x = torch.relu(x)
        x = self.conv8(x)
        if self.bn:
            x = self.bn8(x)
        x = torch.relu(x)
        x = self.conv9(x)
        if self.bn:
            x = self.bn9(x)
        x = torch.relu(x + shortcut)

        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x