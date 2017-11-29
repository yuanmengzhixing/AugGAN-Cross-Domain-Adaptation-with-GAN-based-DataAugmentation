from models import *
import numpy as np
import tqdm
from utils.image_pool import ImagePool
from collections import OrderedDict
from torch.autograd import Variable
import torch
import utils.util as util
import os
import itertools
from utils.logger import Logger


class GANModel():

    def __init__(
            self, root_dir, input_nc, output_nc, ndf, which_direction, pool_size,
            saving_freq, use_sigmoid, name, batch_size, model_dict_path):
        super(GANModel, self).__init__()
        self.pool_size = pool_size
        self.root_dir = root_dir
        self.which_direction = which_direction
        self.display_freq = display_freq
        self.name = name
        self.lr = 0.00002
        self.niter = 75
        self.niter_decay = 100
        self.old_lr = self.lr
        self.saving_freq = saving_freq

        self.Tensor = torch.cuda.FloatTensor

        self.input_A = self.Tensor(batch_size, 3, 256, 256)
        self.input_B = self.Tensor(batch_size, 3, 256, 256)

        self.tf_summary = Logger('./logs', name)
        self.netG_A = define_G(input_nc=input_nc, output_nc=output_nc, ngf=64, seg_channel=6,
                                        which_model_netG='resnet_9blocks', use_dropout=False).cuda()
        self.load_model(model_dict_path)
        for param in self.netG_A.decode_down_1.parameters():
            param.requires_grad = False
        for param in self.netG_A.decode_down_2.parameters():
            param.requires_grad = False
        for param in self.netG_A.segmen.parameters():
            param.requires_grad = False
        
        self.netG_B = OGResnetGenerator(input_nc=input_nc, output_nc=output_nc,
                                        ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9).cuda()

        self.netD_A = define_D(input_nc=input_nc, ndf=ndf,
                               which_model_netD='n_layers', gpu_ids=[0])
        self.netD_B = define_D(input_nc=input_nc, ndf=ndf,
                               which_model_netD='n_layers', gpu_ids=[0])

        self.criterionGAN = GANLoss()
        self.criterionCyC = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        self.optimizer_G = torch.optim.Adam(itertools.chain(
            filter(lambda p: p.requires_grad, self.netG_A.parameters()), self.netG_B.parameters()), lr=0.00002, betas=(0.5, 0.999))
        self.optimizer_D_A = torch.optim.Adam(
            self.netD_A.parameters(), lr=0.00002, betas=(0.5, 0.999))
        self.optimizer_D_B = torch.optim.Adam(
            self.netD_B.parameters(), lr=0.00002, betas=(0.5, 0.999))

        self.lambda_A = 10.0
        self.lambda_B = 7.0
        self.lambda_idt = 0.0

    def train(self, data_loader, epochs):
        self.fake_pool_A = ImagePool(self.pool_size)
        self.fake_pool_B = ImagePool(self.pool_size)

        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            self.folder_path = '../gan-for-synthia/output/{}/epoch_{}'.format(
                self.name, epoch)
            self.save_img_folder_path = os.path.join(
                self.folder_path, 'output_images')

            progress = tqdm.tqdm(data_loader)
            hist = EpochHistory(length=len(progress))
            for data in progress:
                self.set_input(data)
                loss = self.optimize_params()
                hist.add(loss)
                progress.set_description('Epoch #%d' % self.epoch)
                if self.lambda_idt > 0.0:
                    progress.set_postfix(
                        G_A_loss='%.04f' % loss.get('G_A'), G_B_loss='%.04f' % loss.get('G_B'),
                        G_A_cyc_loss='%.04f' % loss.get('Cyc_A'), G_B_cyc_loss='%.04f' % loss.get('Cyc_B'),
                        D_A_loss='%.04f' % loss.get('D_A'), D_B_loss='%.04f' % loss.get('D_B'))
                else:
                    progress.set_postfix(
                        G_A_loss='%.04f' % loss.get('G_A'), G_B_loss='%.04f' % loss.get('G_B'),
                        G_A_cyc_loss='%.04f' % loss.get('Cyc_A'), G_B_cyc_loss='%.04f' % loss.get('Cyc_B'),
                        D_A_loss='%.04f' % loss.get('D_A'), D_B_loss='%.04f' % loss.get('D_B'))

            metrics = hist.metric()
            print('---> Epoch# %d summary loss G_A:{G_A:.4f}, G_B:{G_B:.4f}'
                  'Cyc_A:{Cyc_A:.4f}, Cyc_B:{Cyc_B:.4f}, D_A:{D_A:.4f}, D_B:{D_B:.4f}'
                  .format(self.epoch, **metrics))

            self.summary_image('real_A', self.real_A.data)
            self.summary_image('transferred_A', self.fake_B.data)
            self.summary_image('rec_A', self.rec_A.data)
            self.summary_image('rec_B', self.rec_B.data)

            for k, v in metrics.items():
                self.tf_summary.scalar(k, v, self.epoch)
            if self.epoch % self.saving_freq == 0:
                self.save()
            if epoch > self.niter:
                self.update_learning_rate()

    def set_input(self, data):
        AtoB = self.which_direction == 'AtoB'
        input_A = data.get('A') if AtoB else data.get('B')
        input_B = data.get('B') if AtoB else data.get('A')

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.fake_A = self.netG_B(self.real_B)
        self.fake_B, _, _, _ = self.netG_A(self.real_A)

        self.image_paths = data.get('A_Paths') if AtoB else data.get('B_Paths')

    def optimize_params(self):
#		print('real_A: {}, real_B: {}'.format(type(self.real_A), type(self.real_B)))
#Optimizer Generator
        self.optimizer_G.zero_grad()
        loss = self.backward_generator()
        self.optimizer_G.step()
#Optimizer Discriminator A
        self.optimizer_D_A.zero_grad()
        loss_d_a = self.backward_Discriminator_A()
        self.optimizer_D_A.step()
#Optimizer Discriminator B
        self.optimizer_D_B.zero_grad()
        loss_d_b = self.backward_Discriminator_B()
        self.optimizer_D_B.step()

        loss['D_A'] = loss_d_a.data[0]
        loss['D_B'] = loss_d_b.data[0]

        return loss

    def backward_generator(self):
        if self.lambda_idt > 0:
            idt_A = self.netG_A(self.real_B)
            loss_idt_A = self.criterionIdt(
                idt_A, self.real_B) * self.lambda_A * self.lambda_idt

            idt_B = self.netG_B(self.real_A)
            loss_idt_B = self.criterionIdt(
                idt_B, self.real_A) * self.lambda_B * self.lambda_idt
        else:
            loss_idt_A = 0
            loss_idt_B = 0

        fake_B = self.fake_B
        pred_fake = self.netD_A.forward(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        fake_A = self.fake_A
        pred_fake = self.netD_B.forward(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        self.rec_A = self.netG_B.forward(fake_B)
        loss_cycle_A = self.criterionCyC(self.rec_A, self.real_A) * self.lambda_A
        self.rec_B, _, _, _ = self.netG_A.forward(fake_A)
        loss_cycle_B = self.criterionCyC(self.rec_B, self.real_B) * self.lambda_B

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
        loss_G.backward()

        G_A = loss_G_A.data[0]
        Cyc_A = loss_cycle_A.data[0]
        G_B = loss_G_B.data[0]
        Cyc_B = loss_cycle_B.data[0]

        if self.lambda_idt > 0.0:
            idt_A = loss_idt_A.data[0]
            idt_B = loss_idt_B.data[0]
            return OrderedDict([('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def backward_D_basic(self, netD, real, fake):
		#Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
	#Fake
        pred_fake = netD.forward(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake) / 2
        loss_D.backward()
        return loss_D

    def backward_Discriminator_A(self):
        fake_B = self.fake_pool_B.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        return loss_D_A

    def backward_Discriminator_B(self):
        fake_A = self.fake_pool_A.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_A, self.real_A, fake_A)
        return loss_D_B

    def get_current_error(self):
        D_A = self.loss_D_A.data[0]
        G_A = self.loss_G_A.data[0]
        Cyc_A = self.loss_cycle_A.data[0]
        D_B = self.loss_D_B.data[0]
        G_B = self.loss_G_B.data[0]
        Cyc_B = self.loss_cycle_B.data[0]
        if self.lambda_idt > 0.0:
            idt_A = self.loss_idt_A.data[0]
            idt_B = self.loss_idt_B.data[0]
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A), ('idt_A', idt_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B), ('idt_B', idt_B)])
        else:
            return OrderedDict([('D_A', D_A), ('G_A', G_A), ('Cyc_A', Cyc_A),
                                ('D_B', D_B), ('G_B', G_B), ('Cyc_B', Cyc_B)])

    def load_model(self, model_dict_path):
        self.netG_A.load_state_dict(torch.load(model_dict_path))
        param_dd1 = self.netG_A.decode_down_1.parameters()
        for param in param_dd1:
            param.requires_grad = False
        param_dd2 = self.netG_A.decode_down_2.parameters()
        for param in param_dd2:
            param.requires_grad = False

    def summary_image(self, tag, output):
        def to_numpy(imgs):
            return imgs.squeeze().cpu().numpy()
        imgs = []
        for i, img in enumerate(output):
            imgs.append(to_numpy(img))
        self.tf_summary.image(tag, imgs, self.epoch)

    def save(self):
        util.mkdirs(self.folder_path)
        self.save_network(self.netG_A, 'G_A', self.folder_path)
        self.save_network(self.netG_B, 'G_B', self.folder_path)

    def save_network(self, network, network_label, folder_path):
        save_filename = '%s_net.pth' % network_label
        save_path = os.path.join(folder_path, save_filename)
        torch.save(network.state_dict(), save_path)

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        rec_A = util.tensor2im(self.rec_A.data)
        real_B = util.tensor2im(self.real_B.data)
        fake_A = util.tensor2im(self.fake_A.data)
        rec_B = util.tensor2im(self.rec_B.data)
        if self.lambda_idt > 0.0:
            idt_A = util.tensor2im(self.idt_A.data)
            idt_B = util.tensor2im(self.idt_B.data)
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A), ('idt_B', idt_B),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B), ('idt_A', idt_A)])
        else:
            return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])

    def update_learning_rate(self):
        lrd = self.lr / self.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D_A.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_D_B.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f --> %f' % (self.old_lr, lr))
        self.old_lr = lr


class EpochHistory():

	def __init__(self, length):
		self.count = 0
		self.len = length
		self.loss_term = {'G_A': None, 'G_B': None, 'Cyc_A': None,
                    'Cyc_B': None, 'D_A': None, 'D_B': None}
		self.losses_G_A = np.zeros(self.len)
		self.losses_G_B = np.zeros(self.len)
		self.losses_G_Cyc_A = np.zeros(self.len)
		self.losses_G_Cyc_B = np.zeros(self.len)
		self.losses_D_A = np.zeros(self.len)
		self.losses_D_B = np.zeros(self.len)

	def add(self, loss_dict):
		self.losses_G_A[self.count] = loss_dict.get('G_A')
		self.losses_G_B[self.count] = loss_dict.get('G_B')
		self.losses_G_Cyc_A[self.count] = loss_dict.get('Cyc_A')
		self.losses_G_Cyc_B[self.count] = loss_dict.get('Cyc_B')
		self.losses_D_A[self.count] = loss_dict.get('D_A')
		self.losses_D_B[self.count] = loss_dict.get('D_B')

		for k, v in loss_dict.items():
			if self.loss_term[k] is None:
				self.loss_term[k] = np.zeros(self.len)
			self.loss_term[k][self.count] = v
		self.count += 1

	def metric(self, prefix=''):
		terms = {
			prefix + 'G_A': self.losses_G_A.mean(),
			prefix + 'G_B': self.losses_G_B.mean(),
			prefix + 'Cyc_A': self.losses_G_Cyc_A.mean(),
			prefix + 'Cyc_B': self.losses_G_Cyc_B.mean(),
                    prefix + 'D_A': self.losses_D_A.mean(),
		    prefix + 'D_B': self.losses_D_B.mean(),
		}

		terms.update({
			prefix + k: v.mean() for k, v in self.loss_term.items()
			if v is not None
		})
		return terms


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.cuda.FloatTensor):
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
