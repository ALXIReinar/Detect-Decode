from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

from ml.logger_config import log_event


class CRNNWordDecoder(nn.Module):
    def __init__(
            self,
            num_classes=45, hidden_size=256, num_lstm_layers=2, lstm_dropout=0.3,
            pretrained_backbone=False, use_feature_compressor=True, compressor_output_size=512,
    ):
        """
        CRNN модель для распознавания рукописного текста

        Args:
            num_classes: количество классов (45 для упрощённого алфавита)
            hidden_size: размер скрытого состояния RNN/LSTM
            num_lstm_layers: количество слоёв BiLSTM (рекомендуется 2-3)
            lstm_dropout: регуляризация модели. 0.0-1.0. Устанавливает вероятность отключения нейронов в слое при обработке тензора
            pretrained_backbone: При True подгружает веса для ResNet34
            use_feature_compressor: Использовать Linear слой для сжатия признаков перед BiLSTM (default: True)
            compressor_output_size: Размер выхода feature compressor (default: 512)
        """
        super().__init__()

        self.pretrained_backbone = pretrained_backbone
        self.use_feature_compressor = use_feature_compressor
        self.compressor_output_size = compressor_output_size

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

        # Feature compressor: сжимает 2048 -> 512 для ускорения BiLSTM и избежания overfitting
        # Backbone output: [batch, 128, 16, W] -> flatten -> [batch, W, 2048]
        # После compressor: [batch, W, 512]
        if use_feature_compressor:

            self.feature_compressor_linear = nn.Linear(2048, compressor_output_size, bias=False)
            self.feature_compressor_bn = nn.BatchNorm1d(compressor_output_size)
            self.feature_compressor_silu = nn.SiLU(inplace=True)

            lstm_input_size = compressor_output_size
        else:
            self.feature_compressor = None
            lstm_input_size = 2048

        self.bilstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
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

        # Сжимаем признаки перед BiLSTM (если включено)
        if self.use_feature_compressor:
            x = self.feature_compressor_linear(x)  # [batch, W, 2048] → [batch, W, 512]

            # BatchNorm1d ожидает [batch, features, length]
            x = x.permute(0, 2, 1)  # [batch, 512, W]
            x = self.feature_compressor_bn(x)
            x = x.permute(0, 2, 1)  # [batch, W, 512]

            x = self.feature_compressor_silu(x)

        x, _ = self.bilstm(x)  # [batch, W, hidden_size*2]
        x = self.fc(x)  # [batch, W, num_classes]

        return x.permute(1, 0, 2)  # [W, batch, num_classes] для CTCLoss



model_word_decoder_code = '''
from torch import nn
from torchvision.models import resnet34, ResNet34_Weights

from ml.logger_config import log_event


class CRNNWordDecoder(nn.Module):
    def __init__(
            self,
            num_classes=45, hidden_size=256, num_lstm_layers=2, lstm_dropout=0.3,
            pretrained_backbone=False, use_feature_compressor=True, compressor_output_size=512,
    ):
        """
        CRNN модель для распознавания рукописного текста

        Args:
            num_classes: количество классов (45 для упрощённого алфавита)
            hidden_size: размер скрытого состояния RNN/LSTM
            num_lstm_layers: количество слоёв BiLSTM (рекомендуется 2-3)
            lstm_dropout: регуляризация модели. 0.0-1.0. Устанавливает вероятность отключения нейронов в слое при обработке тензора
            pretrained_backbone: При True подгружает веса для ResNet34
            use_feature_compressor: Использовать Linear слой для сжатия признаков перед BiLSTM (default: True)
            compressor_output_size: Размер выхода feature compressor (default: 512)
        """
        super().__init__()

        self.pretrained_backbone = pretrained_backbone
        self.use_feature_compressor = use_feature_compressor
        self.compressor_output_size = compressor_output_size

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

        # Feature compressor: сжимает 2048 -> 512 для ускорения BiLSTM и избежания overfitting
        # Backbone output: [batch, 128, 16, W] -> flatten -> [batch, W, 2048]
        # После compressor: [batch, W, 512]
        if use_feature_compressor:
            
            self.feature_compressor_linear = nn.Linear(2048, compressor_output_size, bias=False)
            self.feature_compressor_bn = nn.BatchNorm1d(compressor_output_size)
            self.feature_compressor_silu = nn.SiLU(inplace=True)
            
            lstm_input_size = compressor_output_size
        else:
            self.feature_compressor = None
            lstm_input_size = 2048

        self.bilstm = nn.LSTM(
            input_size=lstm_input_size, hidden_size=hidden_size, num_layers=num_lstm_layers, batch_first=True, bidirectional=True,
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

        # Сжимаем признаки перед BiLSTM (если включено)
        if self.use_feature_compressor:
            x = self.feature_compressor_linear(x)  # [batch, W, 2048] → [batch, W, 512]

            # BatchNorm1d ожидает [batch, features, length]
            x = x.permute(0, 2, 1)  # [batch, 512, W]
            x = self.feature_compressor_bn(x)
            x = x.permute(0, 2, 1)  # [batch, W, 512]
            
            x = self.feature_compressor_silu(x) 

        x, _ = self.bilstm(x)  # [batch, W, hidden_size*2]
        x = self.fc(x)  # [batch, W, num_classes]

        return x.permute(1, 0, 2)  # [W, batch, num_classes] для CTCLoss
'''

# model = CRNNWordDecoder(12, 256, 3, 0.3, True, True, 512)
# tsr = torch.rand((1, 3, 64, 384), dtype=torch.float32)
# model(tsr)