import numpy as np
import tqdm
from utils.image_pool import ImagePool
from collections import OrderedDict
from torch.autograd import Variable
import utils.util as util
import os
import itertools
from utils.logger import Logger
import torch.nn.functional as F
import cv2
from models import *

class GANModel():
    def __init__(
		self, root_dir, input_nc, output_nc, ndf, which_direction, pool_size, 
	        saving_freq, use_sigmoid, name, batch_size, bg_weight):
        super(GANModel, self).__init__()
        self.pool_size = pool_size
        self.root_dir = root_dir
        self.which_direction = which_direction
        self.saving_freq = saving_freq
        self.name = name
        self.lr = 0.0002
        self.niter = 110
        self.niter_decay = 100
        self.old_lr = self.lr
        self.save_freq = 4

        self.Tensor = torch.cuda.FloatTensor

        self.input_A = self.Tensor(batch_size, 3, 256, 256)
        self.input_B = self.Tensor(batch_size, 3, 256, 256) 

        self.tf_summary = Logger('./logs', name)

        self.netG_A = define_G(input_nc=input_nc, output_nc=output_nc, seg_channel=6, ngf=64, which_model_netG='multitask_DRN_9blocks', use_dropout=False).cuda()
        self.netG_B = define_G(input_nc=input_nc, output_nc=output_nc, seg_channel=False, ngf=64, which_model_netG='resnet_9blocks',norm='instance', use_dropout=False).cuda()
        
        self.netD_A = define_D(input_nc=3, ndf=ndf, which_model_netD='n_layers', gpu_ids=[0])
        self.netD_B = define_D(input_nc=3, ndf=ndf, which_model_netD='n_layers', gpu_ids=[0])

        self.criterionGAN = GANLoss()
        self.criterionCyC = torch.nn.L1Loss()
        self.criterionIdt = torch.nn.L1Loss()
        ignore_classes = torch.LongTensor([0])
        weight = torch.ones(6)
        weight[ignore_classes] = bg_weight
        self.weight_seg = weight
        print('Segmentation Weights: {}'.format(self.weight_seg))
        self.criterionSeg = Segmentation_Loss(weights=self.weight_seg)
        self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters()), lr=0.0002, betas=(0.5,0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=0.0002, betas=(0.5,0.999))
        self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=0.0002, betas=(0.5,0.999))

        self.lambda_A = 10.0
        self.lambda_B = 7.0
        self.lambda_seg = 0.5

    def train(self, data_loader, epochs):
        self.fake_pool_A = ImagePool(self.pool_size)
        self.fake_pool_B = ImagePool(self.pool_size)

        for epoch in range(1, epochs+1):
            self.epoch = epoch
            self.folder_path = '../gan-for-synthia/output/{}/epoch_{}'.format(self.name, epoch)
            self.save_img_folder_path = os.path.join(self.folder_path, 'output_images')	

            progress = tqdm.tqdm(data_loader)
            hist = EpochHistory(length=len(progress))
            for data in progress:
                self.set_input(data)
                loss = self.optimize_params()
                hist.add(loss)
                progress.set_description('Epoch #%d' % self.epoch)
                progress.set_postfix(
                    G_A_loss='%.04f' % loss.get('G_A'), G_B_loss='%.04f' % loss.get('G_B'),
                    G_A_Seg_loss='%.04f' % loss.get('Seg_A'),
                    D_A_loss='%.04f' % loss.get('D_A'), D_B_loss='%.04f' % loss.get('D_B'),
                    Sim_loss='%.04f' % loss.get('Sim'))
                            
            metrics = hist.metric()
            print('---> Epoch# %d summary loss G_A:{G_A:.4f}, G_B:{G_B:.4f}, '
                    'Seg_A:{Seg_A:.4f}, D_A:{D_A:.4f}, D_B:{D_B:.4f}, Sim:{Sim:.4f}'
                  .format(self.epoch, **metrics))

            
            self.summary_image('real_A', self.real_A.data)
            self.summary_image('real_B', self.real_B.data)
            self.summary_image('transferred_A', self.fake_B.data)
            _, fake_B_seg = torch.max(self.fake_B_seg, 1)
            self.summary_image('Seg_A', fake_B_seg.data)
            self.summary_image('rec_A', self.rec_A.data)
            self.summary_image('rec_B', self.rec_B.data)

            for k,v in metrics.items():
                self.tf_summary.scalar(k,v, self.epoch)
            if self.epoch % self.saving_freq == 0:
                self.save()
            if epoch > self.niter:
                self.update_learning_rate()

    def set_input(self, data):
        AtoB = self.which_direction == 'AtoB'
        input_A = data.get('A') if AtoB else data.get('B')
        input_B = data.get('B') if AtoB else data.get('A')
        semantic_A = data.get('A_label') if AtoB else data.get('label_B')

        self.input_A.resize_(input_A.size()).copy_(input_A)
        self.input_B.resize_(input_B.size()).copy_(input_B)

        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.target_A = Variable(semantic_A).cuda()

        self.fake_A = self.netG_B(self.real_B)
        self.fake_B, self.fake_B_seg, self.cos_sim1, self.cos_sim2 = self.netG_A(self.real_A)
        self.image_paths = data.get('A_Paths') if AtoB else data.get('B_Paths')

    def optimize_params(self):
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

        fake_B = self.fake_B
        fake_B_seg = self.fake_B_seg
        target_A = self.target_A
        pred_fake = self.netD_A.forward(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)
        loss_A_seg, _ = self.criterionSeg(fake_B_seg, target_A) 

        fake_A = self.fake_A
        pred_fake = self.netD_B.forward(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        self.rec_A = self.netG_B.forward(fake_B)
        loss_cycle_A = self.criterionCyC(self.rec_A, self.real_A) * self.lambda_A        
        self.rec_B, _, _, _ = self.netG_A.forward(fake_A)
        loss_cycle_B = self.criterionCyC(self.rec_B, self.real_B) * self.lambda_B

        cos_sim1 = self.cos_sim1
        cos_sim2 = self.cos_sim2

        sim_1 = self.criterionGAN(cos_sim1, False)
        sim_2 = self.criterionGAN(cos_sim2, False)
        sim_loss = sim_1 + sim_2

        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_A_seg * self.lambda_seg
        loss_G.backward()

        G_A = loss_G_A.data[0]
        Cyc_A = loss_cycle_A.data[0]
        G_B = loss_G_B.data[0]
        Cyc_B = loss_cycle_B.data[0]
        Seg_A = loss_A_seg.data[0]
        Sim = sim_loss.data[0]

        return OrderedDict([('G_A', G_A), ('Cyc_A', Cyc_A), ('Seg_A', Seg_A),
                            ('G_B', G_B), ('Cyc_B', Cyc_B), ('Sim', Sim)])

	
    def backward_D_basic(self, netD, real, fake):
		#Real
        pred_real = netD.forward(real)
        loss_D_real = self.criterionGAN(pred_real, True)
		#Fake
        pred_fake = netD.forward(fake)
        loss_D_fake = self.criterionGAN(pred_fake, False)

        loss_D = (loss_D_real + loss_D_fake)/2
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
        self.save_network(self.netD_A, 'D_A', self.folder_path)
        self.save_network(self.netG_B, 'G_B', self.folder_path)
        self.save_network(self.netD_B, 'D_B', self.folder_path)
    
    def save_network(self, network, network_label, folder_path):
        save_filename = '%s_net.pth' % network_label
        save_path = os.path.join(folder_path, save_filename)
        torch.save(network.state_dict(), save_path)
	
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
        self.loss_term = {'G_A':None, 'G_B':None, 'Cyc_A':None, 'Cyc_B':None, 'D_A':None, 'D_B':None, 'Seg_A':None, 'Seg_B':None, 'Sim':None}
        self.losses_G_A = np.zeros(self.len)
        self.losses_G_B = np.zeros(self.len)
        self.losses_G_Cyc_A = np.zeros(self.len)
        self.losses_G_Cyc_B = np.zeros(self.len)
        self.losses_Seg_A = np.zeros(self.len)
#        self.losses_Seg_B = np.zeros(self.len)
        self.losses_D_A = np.zeros(self.len)
        self.losses_D_B = np.zeros(self.len)
        self.losses_Sim = np.zeros(self.len)
    def add(self, loss_dict):
        self.losses_G_A[self.count] = loss_dict.get('G_A')
        self.losses_G_B[self.count] = loss_dict.get('G_B')
        self.losses_G_Cyc_A[self.count] = loss_dict.get('Cyc_A')
        self.losses_G_Cyc_B[self.count] = loss_dict.get('Cyc_B')
        self.losses_Seg_A[self.count] = loss_dict.get('Seg_A')
#        self.losses_Seg_B[self.count] = loss_dict.get('Seg_B')
        self.losses_D_A[self.count] = loss_dict.get('D_A')
        self.losses_D_B[self.count] = loss_dict.get('D_B')
        
        for k,v in loss_dict.items():
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
            prefix + 'Seg_A': self.losses_Seg_A.mean(),
#           prefix + 'Seg_B': self.losses_Seg_B.mean(),
		    prefix + 'D_A': self.losses_D_A.mean(),
		    prefix + 'D_B': self.losses_D_B.mean(),
                    prefix + 'Sim': self.losses_Sim.mean()
		}	
        terms.update({
			prefix + k:v.mean() for k,v in self.loss_term.items()
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
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)

class Segmentation_Loss():
    
	def __init__(self, l1_portion=0.0, weights=None):
		self.l1_criterion = nn.L1Loss().cuda()
		self.crossentropy = nn.CrossEntropyLoss(weight=weights).cuda()
		self.l1_portion = l1_portion

	def __call__(self, pred, target) -> (float,dict):
		pixelwise_loss, loss_term = self.pixelwise_loss(pred, target)

		return pixelwise_loss, loss_term

	def pixelwise_loss(self, pred, target):
		log_pred = F.log_softmax(pred)
		xent_loss = self.crossentropy(log_pred, target)

		if not self.l1_portion:
			return xent_loss, {'xent': xent_loss}

		onehot_target = (
			torch.FloatTensor(pred.size())
			.zero_().cuda()
			.scatter_(1, target.data.unsqueeze(1),1))

		l1_loss = self.l1_criterion(pred, Variable(onehot_target))

		return xent_loss + self.l1_portion*l1_loss, {'xent':xent_loss, 'l1':l1_loss}

	def set_summary_loagger(self, tf_summary):
		self.tf_summary = tf_summary


