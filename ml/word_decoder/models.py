import torch
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

from ml.logger_config import log_event


class CRNNWordEncoder(nn.Module):
    def __init__(self, num_classes=45, hidden_size=256, num_lstm_layers=2, lstm_dropout=0.3, pretrained_backbone=False):
        """
        CRNN модель для распознавания рукописного текста
        
        Args:
            num_classes: количество классов (45 для упрощённого алфавита)
            hidden_size: размер скрытого состояния RNN/LSTM
            num_lstm_layers: количество слоёв BiLSTM (рекомендуется 2-3)
            lstm_dropout: регуляризация модели. 0.0-1.0. Устанавливает вероятность отключения нейронов в слое при обработке тензора
            pretrained_backbone: При True подгружает веса для ResNet34
        """
        super().__init__()
        
        self.pretrained_backbone = pretrained_backbone
        
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None)

        "Бэкбон от ResNet34"
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.layer1,  # 64 channels
            resnet.layer2   # 128 channels
        )
        
        "Замораживаем backbone если pretrained веса"
        if pretrained_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        "Сохраняем ссылки на отдельные слои для постепенного размораживания"
        self.inp_conv = self.backbone[0]
        self.inp_bn = self.backbone[1]
        self.inp_relu = self.backbone[2]
        self.layer1 = self.backbone[3]
        self.layer2 = self.backbone[4]
        
        self.bilstm = nn.LSTM(
            input_size=2048, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0
        )
        
        # *2 потому что bidirectional LSTM выдаёт конкатенацию forward и backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    

    def unfreeze_backbone_gradual(self, stage=1):
        if stage == 1:
            "Размораживаем layer2"
            for param in self.layer2.parameters():
                param.requires_grad = True
            log_event('Разморожен \033[33mlayer2\033[0m', level='WARNING')

        if stage == 2:
            "Размораживаем layer1"
            for param in self.layer1.parameters():
                param.requires_grad = True
            log_event('Разморожен \033[37mlayer1\033[0m', level='WARNING')

        if stage >= 3:
            "Размораживает веса backbone"
            for param in self.backbone.parameters():
                param.requires_grad = True
            log_event('Разморожен \033[36mbackbone\033[0m', level='WARNING')


    def forward(self, x):
        x = self.backbone(x)  # [batch, 128, 16, W]

        b, c, h, w = x.shape

        # Сначала переставляем W на второе место, чтобы схлопывать именно C и H для каждого шага W
        x = x.permute(0, 3, 1, 2).contiguous()  # [batch, W, 128, 16]
        x = x.view(b, w, c * h)  # [batch, W, 2048]

        x, _ = self.bilstm(x)  # [batch, W, hidden_size*2]
        x = self.fc(x)  # [batch, W, num_classes]

        return x.permute(1, 0, 2)  # [W, batch, num_classes] для CTCLoss


model_word_encoder_code = '''
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

from ml.logger_config import log_event


class CRNNWordEncoder(nn.Module):
    def __init__(self, num_classes=45, hidden_size=256, num_lstm_layers=2, lstm_dropout=0.3, pretrained_backbone=False):
        """
        CRNN модель для распознавания рукописного текста
        
        Args:
            num_classes: количество классов (45 для упрощённого алфавита)
            hidden_size: размер скрытого состояния RNN/LSTM
            num_lstm_layers: количество слоёв BiLSTM (рекомендуется 2-3)
            lstm_dropout: регуляризация модели. 0.0-1.0. Устанавливает вероятность отключения нейронов в слое при обработке тензора
            pretrained_backbone: При True подгружает веса для ResNet34
        """
        super().__init__()
        
        self.pretrained_backbone = pretrained_backbone
        
        resnet = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained_backbone else None)

        "Бэкбон от ResNet34"
        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,  # 64 channels
            resnet.layer2   # 128 channels
        )
        
        "Замораживаем backbone если pretrained веса"
        if pretrained_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        "Сохраняем ссылки на отдельные слои для постепенного размораживания"
        self.inp_conv = self.backbone[0]
        self.inp_bn = self.backbone[1]
        self.inp_relu = self.backbone[2]
        self.inp_maxpool = self.backbone[3]
        self.layer1 = self.backbone[4]
        self.layer2 = self.backbone[5]
        
        self.bilstm = nn.LSTM(
            input_size=128, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
            dropout=lstm_dropout if num_lstm_layers > 1 else 0
        )
        
        # *2 потому что bidirectional LSTM выдаёт конкатенацию forward и backward
        self.fc = nn.Linear(hidden_size * 2, num_classes)
    

    def unfreeze_backbone_gradual(self, stage=1):
        if stage == 1:
            "Размораживаем layer2"
            for param in self.layer2.parameters():
                param.requires_grad = True
            log_event('Разморожен \033[33mlayer2\033[0m', level='WARNING')

        if stage == 2:
            "Размораживаем layer1"
            for param in self.layer1.parameters():
                param.requires_grad = True
            log_event('Разморожен \033[37mlayer1\033[0m', level='WARNING')

        if stage >= 3:
            "Размораживает веса backbone"
            for param in self.backbone.parameters():
                param.requires_grad = True
            log_event('Разморожен \033[36mbackbone\033[0m', level='WARNING')


    def forward(self, x):
        x = self.backbone(x)  # [batch, 128, H', W']

        b, c, h, w = x.shape
        x = x.view(b, c * h, w)  # [batch, 128, W]
        x = x.permute(0, 2, 1)  # [batch, W, 128]
        x, _ = self.bilstm(x)  # [batch, W, hidden_size*2]

        # У нас [batch, W, hidden_size*2], Linear применится к каждому из W элементов
        x = self.fc(x)  # [batch, W, num_classes]
        x = x.permute(1, 0, 2)  # [W, batch, num_classes]
        return x
'''