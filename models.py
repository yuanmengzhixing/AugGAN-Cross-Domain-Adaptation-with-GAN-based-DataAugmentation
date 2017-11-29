import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer


def define_G(input_nc, output_nc, seg_channel, ngf, which_model_netG, norm='instance', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'multitask_DRN_9blocks':
        netG = ResnetGenerator(input_nc=input_nc, output_nc=output_nc, seg_channel=seg_channel, ngf=ngf, norm_layer=norm_layer,
                               use_dropout=use_dropout)
    elif which_model_netG == 'resnet_9blocks':
        netG = OGResnetGenerator(input_nc=input_nc, output_nc=output_nc, ngf=64, norm_layer=norm_layer, 
                               use_dropout=False)
    else:
        raise NotImplementedError(
            'Generator model name [%s] is not recognized' % which_model_netG)
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])
    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(
            input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(
            input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(
                    real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(
                    fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/

class ResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, seg_channel, ngf=64, norm_layer=nn.BatchNorm2d, 
                 use_dropout=False, n_blocks=6, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        encoder = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ELU(inplace=True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            encoder += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ELU(inplace=True)]

        mult = 2**n_downsampling
        dilation_metric = [1, 1, 2, 4, 8, 16, 32, 64, 1]
        for i in range(n_blocks):
            dilation = dilation_metric[i]
            encoder += [DilationBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=nn.InstanceNorm2d, use_dropout=use_dropout, use_bias=use_bias, dilation=dilation)]
        self.encoder = nn.Sequential(*encoder)

        self.decode_up_1 = nn.ConvTranspose2d(ngf * 4, int(ngf * 4/2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias)
        self.decode_up_2 = nn.ConvTranspose2d(ngf * 2, int(ngf * 2/2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias)
        self.decode_down_1 = nn.ConvTranspose2d(ngf * 4, int(ngf * 4/2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias)
        self.decode_down_2 = nn.ConvTranspose2d(ngf * 2, int(ngf * 2/2),
                                              kernel_size=3, stride=2,
                                              padding=1, output_padding=1,
                                              bias=use_bias)
        self.norm_1 = norm_layer(int(ngf * 4/2))
        self.norm_2 = norm_layer(int(ngf * 2/2))
        self.elu = nn.ELU(True)
        
        recon = [] 
        recon += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.recon = nn.Sequential(*recon)
        
        segmen = []
        segmen += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, seg_channel, kernel_size=7, padding=0),
            nn.ELU(inplace=True),
            nn.Conv2d(seg_channel, seg_channel, kernel_size=1, stride=1)
        ]
        self.segmen = nn.Sequential(*segmen)
        self.cos_sim = nn.CosineSimilarity(dim=1, eps=1e-6)

    def forward(self, input):
        x = self.encoder(input)

        decode_up = self.decode_up_1(x)
        decode_up = self.norm_1(decode_up)
        decode_up = self.elu(decode_up)
        decode_up = self.decode_up_2(decode_up)
        decode_up = self.norm_2(decode_up)
        decode_up = self.elu(decode_up)

        decode_down = self.decode_down_1(x)
        decode_down = self.norm_1(decode_down)
        decode_down = self.elu(decode_down)
        decode_down = self.decode_down_2(decode_down)
        decode_down = self.norm_2(decode_down)
        decode_down = self.elu(decode_down)
        
        up_dict_1 = self.decode_up_1.state_dict().get('weight')
        up_dict_2 = self.decode_up_2.state_dict().get('weight')
        down_dict_1 = self.decode_down_1.state_dict().get('weight')
        down_dict_2 = self.decode_down_2.state_dict().get('weight')
        
        cos_sim_1 = self.cos_sim(up_dict_1, down_dict_1)
        cos_sim_2 = self.cos_sim(up_dict_2, down_dict_2)

        out_rec = self.recon(decode_up)
        out_seg = self.segmen(decode_down)

        return out_rec, out_seg, cos_sim_1, cos_sim_2

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

## OGResnetGenerator
class OGResnetGenerator(nn.Module):

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(OGResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0,
                           bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

class DilationBlock(nn.Module):

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, dilation):
        super(DilationBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, use_dropout, use_bias, dilation)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias, dilation):
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

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation-1, bias=use_bias, dilation=dilation),
                       norm_layer(dim),
                       nn.ELU(True)]
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
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=dilation-1, bias=use_bias, dilation=dilation),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out




# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids
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
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)
