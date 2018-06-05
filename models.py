import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np

class Encoder(nn.Module):
    def __init__(self, input_nc, nef, norm='batch', use_dropout=False, 
                 n_blocks=3 ,padding_type='reflect'):
        super(Encoder, self).__init__()
        self.input_nc = input_nc
        self.nef = nef
        norm_layer = get_norm_layer(norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        layers = [nn.ReflectionPad2d(3),
                  nn.Conv2d(input_nc, nef, kernel_size=7, padding=0, bias=use_bias),
                  norm_layer(nef),
                  nn.LeakyReLU(inplace=True)]
        n_dsamp = 2
        for i in range(n_dsamp):
            nc_mult = 2**i
            layers += [
                nn.Conv2d(nef * nc_mult, nef * nc_mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(nef * nc_mult * 2),
                nn.LeakyReLU(inplace=True)
            ]
        nc_mult = 2**n_dsamp
        for i in range(n_blocks):
            layers += [
                ResnetBlock(nef * nc_mult, padding_type=padding_type, norm_layer=norm_layer, 
                            use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.mdl = nn.Sequential(*layers)
        self.mdl.apply(weights_init)
    def forward(self, x):
        return self.mdl(x)

class Multitask_Generator(nn.Module):
    def __init__(self, input_nc, output_nc, parse_nc, norm='batch', use_dropout=False, 
                 n_blocks=6, padding_type='reflect'):
        super(Multitask_Generator, self).__init__()
        norm_layer = get_norm_layer(norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d       
        self.shared_x = SharedBlock(input_nc, norm_layer, use_bias, use_dropout, n_blocks, padding_type)
        self.shared_y = SharedBlock(input_nc, norm_layer, use_bias, use_dropout, n_blocks, padding_type)
        self.decoder_x = Decoder(input_nc, output_nc, norm_layer, use_bias)
        self.decoder_y = Decoder(input_nc, parse_nc, norm_layer, use_bias)
    def forward(self, x):
        return self.decoder_x(self.shared_x(x)), self.decoder_y(self.shared_y(x))

class SharedBlock(nn.Module):
    def __init__(self, dim, norm_layer=nn.BatchNorm2d, use_bias=False, use_dropout=False, n_blocks=6, padding_type='reflect'):
        super(SharedBlock, self).__init__()
        self.dim = dim
        layers = []
        for i in range(n_blocks):
            layers += [
                ResnetBlock(dim, padding_type=padding_type, norm_layer=norm_layer,
                            use_dropout=use_dropout, use_bias=use_bias)
            ]
        self.mdl = nn.Sequential(*layers)
        self.mdl.apply(weights_init)
    def forward(self, x):
        return self.mdl(x)

class Decoder(nn.Module):
    def __init__(self, input_nc, output_nc, norm_layer=nn.BatchNorm2d, use_bias=False):
        super(Decoder, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        c_layers = [
            nn.ConvTranspose2d(input_nc, int(input_nc/2), kernel_size=3, stride=2, 
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(input_nc/2)),
            nn.ConvTranspose2d(int(input_nc/2), int(input_nc/4), kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=use_bias),
            norm_layer(int(input_nc/4))
        ]
        ngf = int(input_nc/4)
        ts_layers = []
        if output_nc == 3:
            ts_layers += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.Tanh()
            ]
        else:
            ts_layers += [
                nn.ReflectionPad2d(3),
                nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(output_nc, output_nc, kernel_size=1, stride=1)
            ]
        self.c_layers = nn.Sequential(*c_layers)
        self.c_layers.apply(weights_init)
        self.ts_layers = nn.Sequential(*ts_layers)
        self.ts_layers.apply(weights_init)
    def forward(self, x):
        return self.ts_layers(self.c_layers(x))

# Define a resnet block
# 10/27 tranfer it to dilated block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm='batch', use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        norm_layer = get_norm_layer(norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = int(np.ceil((kw - 1) / 2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)
