from torch.autograd import Variable
import numpy as np
import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
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
        
# Define Segmentation Loss for multitask learning
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

class EpochHistory():
    def __init__(self, length):
        self.count = 0 
        self.len = length
        self.loss_term = {'loss_g_x':None, 'loss_g_y':None, 'loss_cyc_x':None, 'loss_cyc_y':None, 'loss_seg_x':None, 'loss_seg_y':None, 'loss_d_x':None, 'loss_d_y':None, 'loss_ws_x':None, 'loss_ws_y':None}
        self.losses_g_x = np.zeros(self.len)
        self.losses_g_y = np.zeros(self.len)
        self.losses_cyc_x = np.zeros(self.len)
        self.losses_cyc_y = np.zeros(self.len)
        self.losses_seg_x = np.zeros(self.len)
        self.losses_seg_y = np.zeros(self.len)
        self.losses_d_x = np.zeros(self.len)
        self.losses_d_y = np.zeros(self.len)
        self.losses_ws_x = np.zeros(self.len)
        self.losses_ws_y = np.zeros(self.len)
    def add(self, loss_dict):
        self.losses_g_x[self.count] = loss_dict.get('loss_g_x')
        self.losses_g_y[self.count] = loss_dict.get('loss_g_y')
        self.losses_cyc_x[self.count] = loss_dict.get('loss_cyc_x')
        self.losses_cyc_y[self.count] = loss_dict.get('loss_cyc_y')
        self.losses_seg_x[self.count] = loss_dict.get('loss_seg_x')
        self.losses_seg_y[self.count] = loss_dict.get('loss_seg_y')
        self.losses_d_x[self.count] = loss_dict.get('loss_d_x')
        self.losses_d_y[self.count] = loss_dict.get('loss_d_y')
        
        for k,v in loss_dict.items():
            if self.loss_term[k] is None:
                self.loss_term[k] = np.zeros(self.len)
            self.loss_term[k][self.count] = v
        self.count += 1
    def metric(self, prefix=''):
        terms = {
			prefix + 'loss_g_x': self.losses_g_x.mean(),
			prefix + 'loss_g_y': self.losses_g_y.mean(),
			prefix + 'loss_cyc_x': self.losses_cyc_x.mean(),
			prefix + 'loss_cyc_y': self.losses_cyc_y.mean(),
            prefix + 'loss_seg_x': self.losses_seg_x.mean(),
            prefix + 'loss_seg_y': self.losses_seg_y.mean(),
		    prefix + 'loss_d_x': self.losses_d_x.mean(),
		    prefix + 'loss_d_y': self.losses_d_y.mean(),
            prefix + 'loss_ws_x': self.losses_ws_x.mean(),
            prefix + 'loss_ws_y': self.losses_ws_y.mean()
		}	
        terms.update({
			prefix + k:v.mean() for k,v in self.loss_term.items()
			if v is not None
		})
        return terms