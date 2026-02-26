from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class CRNNWordEncoder(nn.Module):
    def __init__(self, num_classes=45, hidden_size=256, num_lstm_layers=2, dropout=0.3, pretrained_backbone=False):
        """
        CRNN модель для распознавания рукописного текста
        
        Args:
            num_classes: количество классов (45 для упрощённого алфавита)
            hidden_size: размер скрытого состояния RNN/LSTM
            num_lstm_layers: количество слоёв BiLSTM (рекомендуется 2-3)
            dropout: регуляризация модели. 0.0-1.0. Устанавливает вероятность отключения нейронов в слое при обработке тензора
        """
        super().__init__()
        
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None)
        
        "Бэкбон от ResNet34"
        self.inp_conv = resnet.conv1
        self.inp_bn = resnet.bn1
        self.inp_relu = resnet.relu
        self.inp_maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        
        self.rnn = nn.RNN(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        
        self.bilstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # *2 потому что bidirectional LSTM выдаёт конкатенацию forward и backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        x = self.inp_conv(x)
        x = self.inp_bn(x)
        x = self.inp_relu(x)
        x = self.inp_maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.mean(dim=2)  # [batch, 128, W]

        x = x.permute(0, 2, 1)  # [batch, W, 128]

        x, _ = self.rnn(x)  # [batch, W, hidden_size]

        x, _ = self.bilstm(x)  # [batch, W, hidden_size*2]

        # У нас [batch, W, hidden_size*2], Linear применится к каждому из W элементов
        x = self.fc(x)  # [batch, W, num_classes]

        x = x.permute(1, 0, 2)  # [W, batch, num_classes]
        return x

model_word_encoder_code = '''
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights


class CRNNWordEncoder(nn.Module):
    def __init__(self, num_classes=45, hidden_size=256, num_lstm_layers=2, dropout=0.3):
        """
        CRNN модель для распознавания рукописного текста
        
        Args:
            num_classes: количество классов (45 для упрощённого алфавита)
            hidden_size: размер скрытого состояния RNN/LSTM
            num_lstm_layers: количество слоёв BiLSTM (рекомендуется 2-3)
            dropout: регуляризация модели. 0.0-1.0. Устанавливает вероятность отключения нейронов в слое при обработке тензора
        """
        super().__init__()
        
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        
        "Бэкбон от ResNet34"
        self.inp_conv = resnet.conv1
        self.inp_bn = resnet.bn1
        self.inp_relu = resnet.relu
        self.inp_maxpool = resnet.maxpool
        
        self.layer1 = resnet.layer1  # 64 channels
        self.layer2 = resnet.layer2  # 128 channels
        
        self.rnn = nn.RNN(input_size=128, hidden_size=hidden_size, num_layers=1, batch_first=True, bidirectional=False)
        
        self.bilstm = nn.LSTM(
            input_size=hidden_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
            dropout=dropout if num_lstm_layers > 1 else 0
        )
        
        # *2 потому что bidirectional LSTM выдаёт конкатенацию forward и backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        
    def forward(self, x):
        x = self.inp_conv(x)
        x = self.inp_bn(x)
        x = self.inp_relu(x)
        x = self.inp_maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)

        x = x.mean(dim=2)  # [batch, 128, W]

        x = x.permute(0, 2, 1)  # [batch, W, 128]

        x, _ = self.rnn(x)  # [batch, W, hidden_size]

        x, _ = self.bilstm(x)  # [batch, W, hidden_size*2]

        # У нас [batch, W, hidden_size*2], Linear применится к каждому из W элементов
        x = self.fc(x)  # [batch, W, num_classes]

        x = x.permute(1, 0, 2)  # [W, batch, num_classes]
        return x
'''