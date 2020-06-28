import shutil
from imutils import paths
import os
import pandas as pd
import numpy as np
import torch
from torchvision import transforms
from PIL import Image
import warnings
from collections import namedtuple
import torch.nn as nn
import torch.nn.functional as F
from torch.jit.annotations import Optional, Tuple
from torch import Tensor
from torch.hub import load_state_dict_from_url

workingDirectory = os.path.split(os.path.dirname(os.path.realpath(__file__)))[0]
covidPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'COVID-19'])
normalPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'NORMAL'])
verificationPath = os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION'])
normalImages = list(paths.list_images(f'{normalPath}'))
covidImages = list(paths.list_images(f'{covidPath}'))
images = []
labels = []
verImg = []
verLabels = []

def processImages(normalImages = normalImages, covidImages = covidImages): #converts image directories to resized greyscale numpy arrays
    global images
    global labels
    global verImg
    global verLabels
    preprocess = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    for i in covidImages:
        label = i.split(os.path.sep)[-2]
        image = Image.open(i).convert('RGB')
        image = preprocess(image)
        images.append(image)
        labels.append(label)
    print('Finished copying COVID-19 images')
    for i in normalImages:
        label = i.split(os.path.sep)[-2]
        image = Image.open(i).convert('RGB')
        image = preprocess(image)
        images.append(image)
        labels.append(label)
    print('Finished copying normal images')
    for (index, row) in pd.read_csv(os.path.sep.join([f'{workingDirectory}', 'verification.csv'])).iterrows():
        verLabels.append(row['finding'])
        image = Image.open(os.path.sep.join([f'{workingDirectory}', 'COVID-19 Radiography Database', 'VERIFICATION', str(row['filename'])])).convert('RGB')
        image = preprocess(image)
        verImg.append(image)
    print('Finished copying verification images')
    print('Number of COVID train files:',str(len(covidImages)))
    print('Number of normal train files:',str(len(normalImages)))
    print('Number of verification images:', str(len(list(paths.list_images(f'{verificationPath}')))))

    labels = [1 if x=='COVID-19' else x for x in labels]
    labels = [0 if x=='NORMAL' else x for x in labels]
    labels = np.asarray(labels)
    verLabels = [1 if x=='COVID-19' else x for x in verLabels]
    verLabels = [0 if x=='normal' else x for x in verLabels]
    verLabels = np.asarray(verLabels)
    images = np.asarray(images)
    verImg = np.asarray(verImg)

__all__ = ['GoogLeNet', 'googlenet', "GoogLeNetOutputs", "_GoogLeNetOutputs"]

model_urls = {
    'googlenet': 'https://download.pytorch.org/models/googlenet-1378be20.pth',
}

GoogLeNetOutputs = namedtuple('GoogLeNetOutputs', ['logits', 'aux_logits2', 'aux_logits1'])
GoogLeNetOutputs.__annotations__ = {'logits': Tensor, 'aux_logits2': Optional[Tensor],
                                    'aux_logits1': Optional[Tensor]}

_GoogLeNetOutputs = GoogLeNetOutputs

def googlenet(pretrained=False, progress=True, **kwargs):
    return GoogLeNet(**kwargs)

class GoogLeNet(nn.Module):
    __constants__ = ['aux_logits', 'transform_input']

    def __init__(self, num_classes=2, aux_logits=True, transform_input=False, init_weights=None,
                blocks=None):
        super(GoogLeNet, self).__init__()
        if blocks is None:
            blocks = [BasicConv2d, Inception, InceptionAux]
        if init_weights is None:
            init_weights = True
        assert len(blocks) == 3
        conv_block = blocks[0]
        inception_block = blocks[1]
        inception_aux_block = blocks[2]

        self.aux_logits = aux_logits
        self.transform_input = transform_input

        self.conv1 = conv_block(3, 64, kernel_size=7, stride=2, padding=3)
        self.maxpool1 = nn.MaxPool2d(3, stride=2, ceil_mode=True)
        self.conv2 = conv_block(64, 64, kernel_size=1)
        self.conv3 = conv_block(64, 192, kernel_size=3, padding=1)
        self.maxpool2 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(2, stride=2, ceil_mode=True)

        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)

        if aux_logits:
            self.aux1 = inception_aux_block(512, num_classes)
            self.aux2 = inception_aux_block(528, num_classes)
        else:
            self.aux1 = None
            self.aux2 = None

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1024, num_classes)

        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                import scipy.stats as stats
                X = stats.truncnorm(-2, 2, scale=0.01)
                values = torch.as_tensor(X.rvs(m.weight.numel()), dtype=m.weight.dtype)
                values = values.view(m.weight.size())
                with torch.no_grad():
                    m.weight.copy_(values)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _transform_input(self, x):
        if self.transform_input:
            x_ch0 = torch.unsqueeze(x[0], 1) * (0.229 / 0.5) + (0.485 - 0.5) / 0.5
            x_ch1 = torch.unsqueeze(x[1], 1) * (0.224 / 0.5) + (0.456 - 0.5) / 0.5
            x_ch2 = torch.unsqueeze(x[2], 1) * (0.225 / 0.5) + (0.406 - 0.5) / 0.5
            x = torch.cat((x_ch0, x_ch1, x_ch2), 1)
        return x

    def _forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux1 is not None:
            if self.training:
                aux1 = self.aux1(x)

        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = torch.jit.annotate(Optional[Tensor], None)
        if self.aux2 is not None:
            if self.training:
                aux2 = self.aux2(x)

        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x, aux2, aux1

    @torch.jit.unused
    def eager_outputs(self, x, aux2, aux1):
        if self.training and self.aux_logits:
            return _GoogLeNetOutputs(x, aux2, aux1)
        else:
            return x

    def forward(self, x):
        x = self._transform_input(x)
        x, aux1, aux2 = self._forward(x)
        aux_defined = self.training and self.aux_logits
        if torch.jit.is_scripting():
            if not aux_defined:
                warnings.warn("Scripted GoogleNet always returns GoogleNetOutputs Tuple")
            return GoogLeNetOutputs(x, aux2, aux1)
        else:
            return self.eager_outputs(x, aux2, aux1)

class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj,
                conv_block=None):
        super(Inception, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.branch1 = conv_block(in_channels, ch1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
            conv_block(in_channels, ch3x3red, kernel_size=1),
            conv_block(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        self.branch3 = nn.Sequential(
            conv_block(in_channels, ch5x5red, kernel_size=1),
            conv_block(ch5x5red, ch5x5, kernel_size=3, padding=1)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1, ceil_mode=True),
            conv_block(in_channels, pool_proj, kernel_size=1)
        )

    def _forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        outputs = [branch1, branch2, branch3, branch4]
        return outputs

    def forward(self, x):
        outputs = self._forward(x)
        return torch.cat(outputs, 1)


class InceptionAux(nn.Module):

    def __init__(self, in_channels, num_classes, conv_block=None):
        super(InceptionAux, self).__init__()
        if conv_block is None:
            conv_block = BasicConv2d
        self.conv = conv_block(in_channels, 128, kernel_size=1)

        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.adaptive_avg_pool2d(x, (4, 4))
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x), inplace=True)
        x = F.dropout(x, 0.7, training=self.training)
        x = self.fc2(x)

        return x

class BasicConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

processImages()

model = googlenet()

train = torch.utils.data.TensorDataset(images, labels)
train = torch.utils.data.DataLoader(train, batch_size = 20)
verification = torch.utils.data.TensorDataset(verImg, verLabels)
verification = torch.utils.data.DataLoader(verification, batch_size = 20)

if torch.cuda.is_available():
    train = train.to('cuda')
    verification = verification.to('cuda')
    model.to('cuda')

for epoch in range(50):
    for xb, yb in train:
        pred = model(xb)
        loss = loss(pred, yb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
    model.eval()
    with torch.no_grad():
        valLoss = sum(loss_func(xb, yb) for xb, yb in validation)
    print('Epoch', epoch, valLoss / len(validation))