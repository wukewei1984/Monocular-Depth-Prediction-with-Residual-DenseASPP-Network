import torch
import torch.nn.functional as F
import math
from torch import nn
from collections import OrderedDict
from torch.nn import BatchNorm2d as bn
import collections


oheight, owidth = 228, 304
class DenseASPP(nn.Module):
    """
    * output_scale can only set as 8 or 16
    """

    '''
        'bn_size': 4,
    'drop_rate': 0,
    'growth_rate': 32,
    'num_init_features': 64,
    'block_config': (6, 12, 24, 16),

    'dropout0': 0.1,
    'dropout1': 0.1,
    'd_feature0': 128,
    'd_feature1': 64,
    '''
#grad_list = []

#def print_grad(grad):
   # grad_list.append(grad)
   # print(grad_list)

    def __init__(self,n_class=19, output_stride=8):
        super(DenseASPP,self).__init__()
        bn_size = 4
        drop_rate = 0
        growth_rate = 16
        num_init_features = 64
        block_config = (6, 12, 24, 16)

        dropout0 = 0.1
        dropout1 = 0.1
        d_feature0 = 128
        d_feature1 = 64

        feature_size = int(output_stride / 8)
        # First convolution
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2, padding=3, bias=False)),
            ('norm0', bn(num_init_features)),
            ('relu0', nn.ReLU(inplace=False)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        # block1*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[0], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 1, block)
        num_features = num_features + block_config[0] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2)
        self.features.add_module('transition%d' % 1, trans)
        num_features = num_features // 2

        # block2*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[1], num_input_features=num_features,
                            bn_size=bn_size, growth_rate=growth_rate, drop_rate=drop_rate)
        self.features.add_module('denseblock%d' % 2, block)
        num_features = num_features + block_config[1] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=feature_size)
        self.features.add_module('transition%d' % 2, trans)
        num_features = num_features // 2

        # block3*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[2], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(2 / feature_size))
        self.features.add_module('denseblock%d' % 3, block)
        num_features = num_features + block_config[2] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 3, trans)
        num_features = num_features // 2

        # block4*****************************************************************************************************
        block = _DenseBlock(num_layers=block_config[3], num_input_features=num_features, bn_size=bn_size,
                            growth_rate=growth_rate, drop_rate=drop_rate, dilation_rate=int(4 / feature_size))
        self.features.add_module('denseblock%d' % 4, block)
        num_features = num_features + block_config[3] * growth_rate

        trans = _Transition(num_input_features=num_features, num_output_features=num_features // 2, stride=1)
        self.features.add_module('transition%d' % 4, trans)
        num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', bn(num_features))
        if feature_size > 1:
            self.features.add_module('upsample', nn.Upsample(scale_factor=2, mode='bilinear'))
        num_features=2048
        self.ASPP_3 = _DenseAsppBlock(input_num=num_features, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=3, drop_out=dropout0, bn_start=False)

        self.ASPP_6 = _DenseAsppBlock(input_num=64, num1=d_feature0, num2=d_feature1,
                                      dilation_rate=6, drop_out=dropout0, bn_start=True)

        self.ASPP_12 = _DenseAsppBlock(input_num=64, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=12, drop_out=dropout0, bn_start=True)

        self.ASPP_18 = _DenseAsppBlock(input_num=64, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=18, drop_out=dropout0, bn_start=True)

        self.ASPP_24 = _DenseAsppBlock(input_num=64, num1=d_feature0, num2=d_feature1,
                                       dilation_rate=24, drop_out=dropout0, bn_start=True)
        num_features = num_features + 5 * d_feature1
        #self.decoder = choose_decoder(decoder)
        #self.ecoder = choose_ecoder(ecoder)
        #self.decoder.apply(weights_init)
        #self.conv3 = nn.Conv2d(832, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv2d(578, 258, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv4 = nn.Conv2d(52, 1, kernel_size=5, stride=2, padding=2, bias=False)
        #self.conv5 = nn.Conv2d(3, 1, kernel_size=3, stride=2, padding=1, bias=False)
        #self.conv6 = nn.Conv2d(1, 1, kernel_size=5, stride=2, padding=1, bias=False)
        #self.conv7 = nn.Conv2d(1, 1, kernel_size=3, stride=2, padding=1, bias=False)

        self.conv11 = nn.Sequential(OrderedDict([
            ('convQ', nn.Conv2d(2368, 1024, kernel_size=1, stride=1, padding=0, bias=False)),
            ('normQ', bn(1024)),
            ('reluQ', nn.ReLU(inplace=False)),
        ]))
        self.conv33 = nn.Sequential(OrderedDict([
            ('convQ', nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1, bias=False)),
            ('normQ', bn(512)),
            ('reluQ', nn.ReLU(inplace=False)),
        ]))
        self.conv111 = nn.Sequential(OrderedDict([
            ('convQ', nn.Conv2d(512, 64, kernel_size=1, stride=1, padding=0, bias=False)),
            ('normQ', bn(64)),
        ]))
        self.conv22 = nn.Sequential(OrderedDict([
            ('convQ', nn.Conv2d(2048, 1024, kernel_size=1, stride=1, padding=0, bias=False)),
            ('normQ', bn(1024)),
            ('reluQ', nn.ReLU(inplace=False)),
        ]))


        self.bilinear = nn.Upsample(size=(oheight, owidth), mode='bilinear')
        self.classification = nn.Sequential(
            nn.Dropout2d(p=dropout1),
            nn.Conv2d(in_channels=num_features, out_channels=n_class, kernel_size=1, padding=0),
            nn.Upsample(scale_factor=8, mode='bilinear'),
        )


        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform(m.weight.data)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, feature):
        x=feature
        #self.conv5 = nn.Conv2d(3, 1, kernel_size=1, stride=1, padding=0, bias=False)
        #x=self.conv5(x)
        #x = self.conv6(x)
        #x = self.conv7(x)
        #feature = self.features(_input)

        #x=feature   #  zhi lu
        #x = self.decoder(x)

        x = self.conv22(x)
        x = self.conv33(x)
        x = self.conv111(x)   #64 32 32


        aspp3 = self.ASPP_3(feature)
        #feature = torch.cat((aspp3, feature), dim=1)
        #featurec = torch.cat((aspp3, feature), dim=1)
        feature = aspp3+ x

        aspp6 = self.ASPP_6(feature)
        #feature = torch.cat((aspp6, feature), dim=1)
        #featurec = torch.cat((aspp6, featurec), dim=1)
        feature = feature+aspp6

        aspp12 = self.ASPP_12(feature)
        #feature = torch.cat((aspp12, feature), dim=1)
        #featurec = torch.cat((aspp12, featurec), dim=1)
        feature = feature + aspp12


        aspp18 = self.ASPP_18(feature)
        #feature = torch.cat((aspp18, feature), dim=1)
        #featurec = torch.cat((aspp18, featurec), dim=1)
        feature = feature + aspp18

        aspp24 = self.ASPP_24(feature)
        #featurec = torch.cat((aspp24, featurec), dim=1)
        #feature = torch.cat((aspp24, feature), dim=1)
        #feature = aspp3+aspp6+aspp12+aspp18+aspp24
        #feature = feature + x
        feature = feature + aspp24
        feature = torch.cat((feature, x), dim=1)




        #x = self.conv22(x)
        #feature = x+feature
        #feature = self.conv11(feature)
        #feature = self.conv33(feature)
        #feature = self.conv111(feature)
        #feature = self.conv3(feature)

        #feature = self.ecoder(feature)

###########################################
        #feature = torch.cat((feature, x), dim=1)

        #feature = self.conv4(feature)
        #cls = self.bilinear(feature)
        #print (featurec.grad)
        #featurec.register_hook(print_grad)

        return feature


class _DenseAsppBlock(nn.Sequential):
    """ ConvNet block for building DenseASPP. """

    def __init__(self, input_num, num1, num2, dilation_rate, drop_out, bn_start=True):
        super(_DenseAsppBlock, self).__init__()
        if bn_start:
            self.add_module('norm1', bn(input_num, momentum=0.0003)),

        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(in_channels=input_num, out_channels=num1, kernel_size=1)),

        self.add_module('norm2', bn(num1, momentum=0.0003)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(in_channels=num1, out_channels=num2, kernel_size=3,
                                            dilation=dilation_rate, padding=dilation_rate)),

        self.drop_rate = drop_out

    def forward(self, _input):
        feature = super(_DenseAsppBlock, self).forward(_input)

        if self.drop_rate > 0:
            feature = F.dropout2d(feature, p=self.drop_rate, training=self.training)

        return feature


class _DenseLayer(nn.Sequential):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, dilation_rate=1):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', bn(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=False)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                        growth_rate, kernel_size=1, stride=1, bias=False)),
        self.add_module('norm2', bn(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=False)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                        kernel_size=3, stride=1, dilation=dilation_rate, padding=dilation_rate, bias=False)),
        self.drop_rate = drop_rate

    def forward(self, x):
        new_features = super(_DenseLayer, self).forward(x)
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return torch.cat([x, new_features], 1)


class _DenseBlock(nn.Sequential):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, dilation_rate=1):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(num_input_features + i * growth_rate, growth_rate,
                                bn_size, drop_rate, dilation_rate=dilation_rate)
            self.add_module('denselayer%d' % (i + 1), layer)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, stride=2):
        super(_Transition, self).__init__()
        self.add_module('norm', bn(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=False))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False))
        if stride == 2:
            self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=stride))

def DenseASPP121(n_classes):
    return DenseASPP(n_classes=n_classes)




#if __name__ == "__main__":
   # model = DenseASPP(2)
    #print(model)
