import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
from torch import Tensor
from torch.nn import Parameter
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
import math



###############################################################################
# Helper Functions
###############################################################################


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)
    init_weights(net, init_type, gain=init_gain)
    return net


def define_G(input_nc, output_nc, ngf, netG, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02, gpu_ids=[], cbam=False):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netG == 'basic':
        net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)
    elif netG == 'set':
        net = ResnetSetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, vgg=False, cbam=cbam)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % netG)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02, gpu_ids=[]):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'set':
        net = NLayerSetDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)
    return init_net(net, init_type, init_gain, gpu_ids)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0, use_WAGAN=False, smooth_labels=False):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.WGAN = use_WAGAN
        self.smooth_labels = smooth_labels  # Add the smooth_labels option
        if use_lsgan:
            self.loss = nn.MSELoss()
        elif self.WGAN:
            self.loss = None
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if self.smooth_labels:
            # Apply label smoothing
            if target_is_real:
                target_tensor = torch.rand_like(input) * 0.1 + 0.9  # Smooth real labels
            else:
                target_tensor = torch.rand_like(input) * 0.1  # Keep fake labels close to 0
        else:
            if target_is_real:
                target_tensor = self.real_label
            else:
                target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        if not self.WGAN:
            loss = self.loss(input, target_tensor)
        else:
            if target_is_real:
                loss = -input.mean()
            else:
                loss = input.mean()
        return loss

   
def cal_gradient_penalty(netD, real_data, fake_data, device, type='mixed', constant=1.0, lambda_gp=10.0):
    if lambda_gp > 0.0:
        if type == 'real':   # either use real images, fake images, or a linear interpolation of two.
            interpolatesv = real_data
        elif type == 'fake':
            interpolatesv = fake_data
        elif type == 'mixed':
            alpha = torch.rand(real_data.shape[0], 1, device=device)
            alpha = alpha.expand(real_data.shape[0], real_data.nelement() // real_data.shape[0]).contiguous().view(*real_data.shape)
            interpolatesv = alpha * real_data + ((1 - alpha) * fake_data)
        else:
            raise NotImplementedError('{} not implemented'.format(type))
        interpolatesv.requires_grad_(True)
        disc_interpolates = netD(interpolatesv)
        gradients = torch.autograd.grad(outputs=disc_interpolates, inputs=interpolatesv,
                                        grad_outputs=torch.ones(disc_interpolates.size()).to(device),
                                        create_graph=True, retain_graph=True, only_inputs=True)
        gradients = gradients[0].view(real_data.size(0), -1)  # flat the data
        gradient_penalty = (((gradients + 1e-16).norm(2, dim=1) - constant) ** 2).mean() * lambda_gp        # added eps
        return gradient_penalty, gradients
    else:
        return 0.0, None

# Define spectral normalization layer
# Code from Christian Cosgrove's repository
# https://github.com/christiancosgrove/pytorch-spectral-normalization-gan/blob/master/spectral_normalization.py
def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)

class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = nn.Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = nn.Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
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
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]
        
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
        return self.model(input)


# ResNet generator for "set" of instance attributes
# See https://openreview.net/forum?id=ryxwJhC9YX for details
class ResnetSetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', vgg=False, cbam=False):
        assert (n_blocks >= 0)
        super(ResnetSetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_CBAM = cbam 
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        n_downsampling = 2
        self.encoder_img = self.get_encoder(input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.encoder_seg = self.get_encoder(1, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias)
        self.decoder_img = self.get_decoder(output_nc, n_downsampling, 2 * ngf, norm_layer, use_bias)  # 2*ngf
        self.decoder_seg = self.get_decoder(1, n_downsampling, 3 * ngf, norm_layer, use_bias)  # 3*ngf
        
    def get_encoder(self, input_nc, n_downsampling, ngf, norm_layer, use_dropout, n_blocks, padding_type, use_bias, vgg=False):
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias)]
        
        model += [norm_layer(ngf)]

        model += [nn.ReLU(True)]

        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias)]
            model += [norm_layer(ngf * mult * 2)]
            model += [nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            if not vgg:
                model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias, cbam=self.use_CBAM)]
            else:
                model += [VGGBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        return nn.Sequential(*model)
    
    def get_decoder(self, output_nc, n_downsampling, ngf, norm_layer, use_bias):
        model = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias)]
            model += [norm_layer(int(ngf * mult / 2))]
            model += [nn.ReLU(True)] 

        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        return nn.Sequential(*model)
    
    def forward(self, inp, resolution=None):
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)
        mean = (segs + 1).mean(0).mean(-1).mean(-1)
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        ## run encoder
        enc_img = self.encoder_img(img)
        # run encoder
        enc_img = self.encoder_img(img)
        enc_segs = [self.encoder_seg(segs[:, i, :, :].unsqueeze(1)) for i in range(segs.size(1)) if mean[i] > 0]

        enc_segs_stack = torch.stack(enc_segs, dim=0)  # Stack along a new dimension
        enc_segs_sum = torch.sum(enc_segs_stack, dim=0)  # Sum along the new dimension

        feat = torch.cat([enc_img, enc_segs_sum], dim=1)
        out = [self.decoder_img(feat)]
        idx = 0
        for i in range(segs.size(1)):
            if mean[i] > 0:
                enc_seg = enc_segs[idx]  # (B, ngf, w, h)
                idx += 1  # move to next index
                feat = torch.cat([enc_seg, enc_img, enc_segs_sum], dim=1)
                out += [self.decoder_seg(feat)]
            else:
                out += [segs[:, i, :, :].unsqueeze(1)]  # skip empty segmentation
        return torch.cat(out, dim=1)
    
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias, cbam=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)
        self.cbam = CBAM(dim) if cbam else None

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
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

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
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]
        return nn.Sequential(*conv_block)

    def forward(self, x):
        #out = x + self.conv_block(x)
        out = self.conv_block(x)
        if self.cbam is not None:
            residual = out.clone()  
            out = self.cbam(out)
            out += residual  
            out = nn.ReLU(True)(out)
        else:
            out += x  
        return out


class VGGBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        super(VGGBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 1  # Set padding to 1 to match ResNet
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 1  # Set padding to 1 to match ResNet
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

    
#defines a CBAM from the paper: https://arxiv.org/pdf/1807.06521.pdf 
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CBAM, self).__init__()
        self.channel = channel
        self.MLP = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(),
            nn.Linear(channel // reduction, channel)
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        avg_pool = F.avg_pool2d(x, x.size(2))
        max_pool = F.max_pool2d(x, x.size(2))
        avg_pool = avg_pool.view(avg_pool.size(0), -1)
        max_pool = max_pool.view(max_pool.size(0), -1)
        channel_avg = self.MLP(avg_pool)
        channel_max = self.MLP(max_pool)
        channel_attention = torch.sigmoid(channel_avg + channel_max).unsqueeze(2).unsqueeze(3).expand_as(x)
        x = x * channel_attention
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_max, _ = torch.max(x, dim=1, keepdim=True)
        spatial_attention = torch.sigmoid(spatial_avg + spatial_max).expand_as(x)
        x = x * spatial_attention
        return x
    

#defines historical averaging as proposed in: https://arxiv.org/pdf/1606.03498.pdf
class HistoricalAveraging:
    def __init__(self, decay=0.99):
        self.decay = decay  # Decay factor for the moving average
        self.values = {}    # Dictionary to store current and running averages

    def register_variable(self, name, variable):
        # Register a variable to track and average
        self.values[name] = {
            'current': variable.clone().detach(),  # Initial value
            'running_avg': variable.clone().detach(),  # Initial value for running average
        }

    def update_variable(self, name, variable):
        # Update the current value of the variable
        self.values[name]['current'] = variable

    def compute_running_average(self, name):
        # Update the running average using exponential moving average
        current = self.values[name]['current']
        running_avg = self.values[name]['running_avg']
        running_avg = (self.decay * running_avg) + ((1 - self.decay) * current)
        self.values[name]['running_avg'] = running_avg

    def get_running_average(self, name):
        # Get the running average of a variable
        return self.values[name]['running_avg']

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # Use spectral normalization
        sequence += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


# PatchGAN discriminator for "set" of instance attributes
# See https://openreview.net/forum?id=ryxwJhC9YX for details
class NLayerSetDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerSetDiscriminator, self).__init__()
        self.input_nc = input_nc
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        self.feature_img = self.get_feature_extractor(input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.feature_seg = self.get_feature_extractor(1, ndf, n_layers, kw, padw, norm_layer, use_bias)
        self.classifier = self.get_classifier(2 * ndf, n_layers, kw, padw, norm_layer, use_sigmoid)  # 2*ndf

    def get_feature_extractor(self, input_nc, ndf, n_layers, kw, padw, norm_layer, use_bias):
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw)),
            nn.LeakyReLU(0.2, True)
        ]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            model += [
                # Use spectral normalization
                SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias)),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        return nn.Sequential(*model)

    def get_classifier(self, ndf, n_layers, kw, padw, norm_layer, use_sigmoid):
        nf_mult_prev = min(2 ** (n_layers-1), 8)
        nf_mult = min(2 ** n_layers, 8)
        model = [
            # Use spectral normalization
            SpectralNorm(nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw)),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        # Use spectral normalization
        model += [SpectralNorm(nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw))]
        if use_sigmoid:
            model += [nn.Sigmoid()]
        return nn.Sequential(*model)

    def forward(self, inp):
        # split data
        img = inp[:, :self.input_nc, :, :]  # (B, CX, W, H)
        segs = inp[:, self.input_nc:, :, :]  # (B, CA, W, H)
        mean = (segs + 1).mean(0).mean(-1).mean(-1)
        if mean.sum() == 0:
            mean[0] = 1  # forward at least one segmentation

        # run feature extractor
        feat_img = self.feature_img(img)
        feat_segs = list()
        for i in range(segs.size(1)):
            if mean[i] > 0:  # skip empty segmentation
                seg = segs[:, i, :, :].unsqueeze(1)
                feat_segs.append(self.feature_seg(seg))
        feat_segs_sum = torch.sum(torch.stack(feat_segs), dim=0)  # aggregated set feature

        # run classifier
        feat = torch.cat([feat_img, feat_segs_sum], dim=1)
        out = self.classifier(feat)
        return out
    
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)


# Define Total Variation Loss
class TVLoss(nn.Module):
    def __init__(self):
        super(TVLoss, self).__init__()

    def forward(self, x):
        h_tv = torch.sum(torch.abs(x[:, :, :, :-1] - x[:, :, :, 1:]))
        v_tv = torch.sum(torch.abs(x[:, :, :-1, :] - x[:, :, 1:, :]))
        return h_tv + v_tv

class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def global_style_statistics(self, input):
        b, c, h, w = input.size()
        features = input.view(b, c, -1)
        mean = torch.mean(features, dim=2, keepdim=True)
        cov = torch.matmul(features - mean, (features - mean).permute(0, 2, 1)) / (h * w)
        return mean, cov

    def forward(self, x, y, mask_x, mask_y):
        # Apply masks to input tensors x and y
        x_masked = x * mask_x
        y_masked = y * mask_y

        # Calculate global style statistics for the masked regions
        mean_x, cov_x = self.global_style_statistics(x_masked)
        mean_y, cov_y = self.global_style_statistics(y_masked)

        # Compute style loss using mean squared difference for mean and covariance
        loss_mean = F.mse_loss(mean_x, mean_y)
        loss_cov = F.mse_loss(cov_x, cov_y)

        # Combine the losses (you can adjust the weights as needed)
        loss = loss_mean + loss_cov
        return loss

