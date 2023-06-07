import torch.nn as nn
import torch.nn.init as init
import torch

# DnCNN网络架构
class DnCNN(nn.Module):
    def __init__(self, depth=17, n_channels=64, image_channels=1, use_bnorm=True, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []

        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))

        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)
        self._initialize_weights()

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.orthogonal_(m.weight)
                print('init weight')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
                
# REDNet10, REDNet20, REDNet30网络
class REDNet10(nn.Module):
    def __init__(self, num_layers=5, num_features=64):
        super(REDNet10, self).__init__()
        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv_layers(x)
        out = self.deconv_layers(out)
        out += residual
        out = self.relu(out)
        return out


class REDNet20(nn.Module):
    def __init__(self, num_layers=10, num_features=64):
        super(REDNet20, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x


class REDNet30(nn.Module):
    def __init__(self, num_layers=15, num_features=64):
        super(REDNet30, self).__init__()
        self.num_layers = num_layers

        conv_layers = []
        deconv_layers = []

        conv_layers.append(nn.Sequential(nn.Conv2d(3, num_features, kernel_size=3, stride=2, padding=1),
                                         nn.ReLU(inplace=True)))
        for i in range(num_layers - 1):
            conv_layers.append(nn.Sequential(nn.Conv2d(num_features, num_features, kernel_size=3, padding=1),
                                             nn.ReLU(inplace=True)))

        for i in range(num_layers - 1):
            deconv_layers.append(nn.Sequential(nn.ConvTranspose2d(num_features, num_features, kernel_size=3, padding=1),
                                               nn.ReLU(inplace=True)))
        deconv_layers.append(nn.ConvTranspose2d(num_features, 3, kernel_size=3, stride=2, padding=1, output_padding=1))

        self.conv_layers = nn.Sequential(*conv_layers)
        self.deconv_layers = nn.Sequential(*deconv_layers)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        conv_feats = []
        for i in range(self.num_layers):
            x = self.conv_layers[i](x)
            if (i + 1) % 2 == 0 and len(conv_feats) < math.ceil(self.num_layers / 2) - 1:
                conv_feats.append(x)

        conv_feats_idx = 0
        for i in range(self.num_layers):
            x = self.deconv_layers[i](x)
            if (i + 1 + self.num_layers) % 2 == 0 and conv_feats_idx < len(conv_feats):
                conv_feat = conv_feats[-(conv_feats_idx + 1)]
                conv_feats_idx += 1
                x = x + conv_feat
                x = self.relu(x)

        x += residual
        x = self.relu(x)

        return x



class ADNet(nn.Module):
    def __init__(self, channels, num_of_layers=15):
        super(ADNet, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        groups =1
        layers = []
        kernel_size1 = 1
        '''
        #self.gamma = nn.Parameter(torch.zeros(1))
        '''
        self.conv1_1 = nn.Sequential(nn.Conv2d(in_channels=channels,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_2 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_3 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_4 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_5 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_6 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_7 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_8 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_9 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_10 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_11 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_12 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=2,groups=groups,bias=False,dilation=2),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_13 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_14 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=padding,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_15 = nn.Sequential(nn.Conv2d(in_channels=features,out_channels=features,kernel_size=kernel_size,padding=1,groups=groups,bias=False),nn.BatchNorm2d(features),nn.ReLU(inplace=True))
        self.conv1_16 = nn.Conv2d(in_channels=features,out_channels=1,kernel_size=kernel_size,padding=1,groups=groups,bias=False)
        self.conv3 = nn.Conv2d(in_channels=2,out_channels=1,kernel_size=1,stride=1,padding=0,groups=1,bias=True)
        self.ReLU = nn.ReLU(inplace=True)
        self.Tanh= nn.Tanh()
        self.sigmoid = nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.normal_(0, (2 / (9.0 * 64)) ** 0.5)
                clip_b = 0.025
                w = m.weight.data.shape[0]
                for j in range(w):
                    if m.weight.data[j] >= 0 and m.weight.data[j] < clip_b:
                        m.weight.data[j] = clip_b
                    elif m.weight.data[j] > -clip_b and m.weight.data[j] < 0:
                        m.weight.data[j] = -clip_b
                m.running_var.fill_(0.01)
    def _make_layers(self, block,features, kernel_size, num_of_layers, padding=1, groups=1, bias=False):
        layers = []
        for _ in range(num_of_layers):
            layers.append(block(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding, groups=groups, bias=bias))
        return nn.Sequential(*layers)
    def forward(self, x):
        input = x
        x1 = self.conv1_1(x)
        x1 = self.conv1_2(x1)
        x1 = self.conv1_3(x1)
        x1 = self.conv1_4(x1)
        x1 = self.conv1_5(x1)
        x1 = self.conv1_6(x1)
        x1 = self.conv1_7(x1)
        x1t = self.conv1_8(x1)
        x1 = self.conv1_9(x1t)
        x1 = self.conv1_10(x1)
        x1 = self.conv1_11(x1)
        x1 = self.conv1_12(x1)
        x1 = self.conv1_13(x1)
        x1 = self.conv1_14(x1)
        x1 = self.conv1_15(x1)
        x1 = self.conv1_16(x1)
        out = torch.cat([x,x1],1)
        out= self.Tanh(out)
        out = self.conv3(out)
        out = out*x1
        out2 = x - out
        return out2
    
    
    import torch
import torch.nn as nn


class ConvLayer1(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer1, self).__init__()
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size-1)//2, stride= stride)

        nn.init.xavier_normal_(self.conv2d.weight.data)

    def forward(self, x):
        # out = self.reflection_pad(x)
        # out = self.conv2d(out)
        return self.conv2d(x)


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        padding = (kernel_size-1)//2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, stride=stride),
            nn.ReLU()
        )
        nn.init.xavier_normal_(self.block[0].weight.data)

    def forward(self, x):
        return self.block(x)


class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x ,y ):
        return torch.mul((1-self.delta), x) + torch.mul(self.delta, y)


class Encoding_block(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Encoding_block, self).__init__()
        self.n_convblock = n_convblock
        modules_body = []
        for i in range(self.n_convblock-1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=2))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        for i in range(self.n_convblock-1):
            x = self.body[i](x)
        ecode = x
        x = self.body[self.n_convblock-1](x)
        return ecode, x


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        self.conv2d = ConvLayer(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = torch.nn.functional.interpolate(x_in, scale_factor=self.upsample)
        out = self.conv2d(x_in)
        return out


class upsample1(nn.Module):
    def __init__(self, base_filter):
        super(upsample1, self).__init__()
        self.conv1 = ConvLayer(base_filter, base_filter, 3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(base_filter, base_filter, kernel_size=3, stride=1, upsample=2)
        self.cat = ConvLayer1(base_filter*2, base_filter, kernel_size=1, stride=1)

    def forward(self, x, y):
        y = self.ConvTranspose(y)
        x = self.conv1(x)
        return self.cat(torch.cat((x, y), dim=1))


class Decoding_block2(nn.Module):
    def __init__(self, base_filter, n_convblock):
        super(Decoding_block2, self).__init__()
        self.n_convblock = n_convblock
        self.upsample = upsample1(base_filter)
        modules_body = []
        for i in range(self.n_convblock-1):
            modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        modules_body.append(ConvLayer(base_filter, base_filter, 3, stride=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x, y):
        x = self.upsample(x, y)
        for i in range(self.n_convblock):
            x = self.body[i](x)
        return x

#Corresponds to DEAM Module in NLO Sub-network
class Attention_unet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(Attention_unet, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )
        self.cat = ConvLayer1(in_channels=channel * 2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.ConvTranspose = UpsampleConvLayer(channel, channel, kernel_size=3, stride=1, upsample=2)#up-sampling

    def forward(self, x, g):
        up_g = self.ConvTranspose(g)
        weight = self.conv_du(self.cat(torch.cat([self.C(x), up_g], 1)))
        rich_x = torch.mul((1 - weight), up_g) + torch.mul(weight, x)
        return rich_x

#Corresponds to NLO Sub-network
class ziwangluo1(nn.Module):
    def __init__(self, base_filter, n_convblock_in, n_convblock_out):
        super(ziwangluo1, self).__init__()
        self.conv_dila1 = ConvLayer1(64, 64, 3, 1)
        self.conv_dila2 = ConvLayer1(64, 64, 5, 1)
        self.conv_dila3 = ConvLayer1(64, 64, 7, 1)

        self.cat1 = torch.nn.Conv2d(in_channels=64 * 3, out_channels=64, kernel_size=1, stride=1, padding=0,
                                    dilation=1, bias=True)
        nn.init.xavier_normal_(self.cat1.weight.data)
        self.e3 = Encoding_block(base_filter, n_convblock_in)
        self.e2 = Encoding_block(base_filter, n_convblock_in)
        self.e1 = Encoding_block(base_filter, n_convblock_in)
        self.e0 = Encoding_block(base_filter, n_convblock_in)


        self.attention3 = Attention_unet(base_filter)
        self.attention2 = Attention_unet(base_filter)
        self.attention1 = Attention_unet(base_filter)
        self.attention0 = Attention_unet(base_filter)
        self.mid = nn.Sequential(ConvLayer(base_filter, base_filter, 3, 1),
                                 ConvLayer(base_filter, base_filter, 3, 1))
        self.de3 = Decoding_block2(base_filter, n_convblock_out)
        self.de2 = Decoding_block2(base_filter, n_convblock_out)
        self.de1 = Decoding_block2(base_filter, n_convblock_out)
        self.de0 = Decoding_block2(base_filter, n_convblock_out)

        self.final = ConvLayer1(base_filter, base_filter, 3, stride=1)

    def forward(self, x):
        _input = x
        encode0, down0 = self.e0(x)
        encode1, down1 = self.e1(down0)
        encode2, down2 = self.e2(down1)
        encode3, down3 = self.e3(down2)

        # media_end = self.Encoding_block_end(down3)
        media_end = self.mid(down3)

        g_conv3 = self.attention3(encode3, media_end)
        up3 = self.de3(g_conv3, media_end)
        g_conv2 = self.attention2(encode2, up3)
        up2 = self.de2(g_conv2, up3)

        g_conv1 = self.attention1(encode1, up2)
        up1 = self.de1(g_conv1, up2)

        g_conv0 = self.attention0(encode0, up1)
        up0 = self.de0(g_conv0, up1)

        final = self.final(up0)

        return _input+final


class line(nn.Module):
    def __init__(self):
        super(line, self).__init__()
        self.delta = nn.Parameter(torch.randn(1, 1))

    def forward(self, x, y):
        return torch.mul((1 - self.delta), x) + torch.mul(self.delta, y)


class SCA(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SCA, self).__init__()
        self.conv_du = nn.Sequential(
                ConvLayer1(in_channels=channel, out_channels=channel // reduction, kernel_size=3, stride=1),
                nn.ReLU(inplace=True),
                ConvLayer1(in_channels=channel // reduction, out_channels=channel, kernel_size=3, stride=1),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.conv_du(x)
        return y


class Weight(nn.Module):
    def __init__(self, channel):
        super(Weight, self).__init__()
        self.cat =ConvLayer1(in_channels=channel*2, out_channels=channel, kernel_size=1, stride=1)
        self.C = ConvLayer1(in_channels=channel, out_channels=channel, kernel_size=3, stride=1)
        self.weight = SCA(channel)

    def forward(self, x, y):
        delta = self.weight(self.cat(torch.cat([self.C(y), x], 1)))
        return delta


class transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1),

        )

    def forward(self, x):
        y = self.ext(x)
        return y+self.pre(y)


class Inverse_transform_function(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Inverse_transform_function, self).__init__()
        self.ext = ConvLayer1(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1)
        self.pre = torch.nn.Sequential(
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            ConvLayer1(in_channels=in_channel, out_channels=in_channel, kernel_size=3, stride=1),
        )

    def forward(self, x):
        x = self.pre(x)+x
        x = self.ext(x)
        return x


class Deam(nn.Module):
    def __init__(self, Isreal):
        super(Deam, self).__init__()
        if Isreal:
            self.transform_function = transform_function(3, 64)
            self.inverse_transform_function = Inverse_transform_function(64, 3)
        else:
            self.transform_function = transform_function(1, 64)
            self.inverse_transform_function = Inverse_transform_function(64, 1)

        self.line11 = Weight(64)
        self.line22 = Weight(64)
        self.line33 = Weight(64)
        self.line44 = Weight(64)

        self.net2 = ziwangluo1(64, 3, 2)

    def forward(self, x):
        x = self.transform_function(x)
        y = x

        # Corresponds to NLO Sub-network
        x1 = self.net2(y)
        # Corresponds to DEAM Module
        delta_1 = self.line11(x1, y)
        x1 = torch.mul((1 - delta_1), x1) + torch.mul(delta_1, y)

        x2 = self.net2(x1)
        delta_2 = self.line22(x2, y)
        x2 = torch.mul((1 - delta_2), x2) + torch.mul(delta_2, y)

        x3 = self.net2(x2)
        delta_3 = self.line33(x3, y)
        x3 = torch.mul((1 - delta_3), x3) + torch.mul(delta_3, y)

        x4 = self.net2(x3)
        delta_4 = self.line44(x4, y)
        x4 = torch.mul((1 - delta_4), x4) + torch.mul(delta_4, y)
        x4 = self.inverse_transform_function(x4)
        return x4


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
