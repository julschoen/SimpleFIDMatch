import torch
import torch.nn.functional as F
import torchvision.utils as vutils
from carbontracker.tracker import CarbonTracker

import os
from fid import fid, get_activations
from mmd import mix_rbf_mmd2
from inception import InceptionV3



class Trainer():
	def __init__(self, params, train_loader):
		self.p = params
		if not os.path.isdir(self.p.log_dir):
			os.mkdir(self.p.log_dir)

		if self.p.mmd:
			self.train_loader = train_loader
			self.gen = self.inf_train_gen()
			self.sigma_list = [1, 2, 4, 8, 16, 24, 32, 64]
			block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[2048]
			self.model = InceptionV3([block_idx]).to(self.p.device)

		if self.p.init_ims:
			self.ims = torch.load('means.pt')
			self.ims = self.ims.to(self.p.device)
		else:
			self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)

		self.labels = torch.arange(10).repeat(self.p.num_ims,1).T.flatten()

		if not os.path.isdir('./cdc_carbon'):
			os.mkdir('./cdc_carbon')
		self.tracker = CarbonTracker(epochs=self.p.niter, log_dir='./cdc_carbon/')

	def inf_train_gen(self):
		while True:
			for data in self.train_loader:
				yield data

	def log_interpolation(self, step, c, ims):
		path = os.path.join(self.p.log_dir, 'images')
		if not os.path.isdir(path):
			os.mkdir(path)
		path = os.path.join(self.p.log_dir, 'images',f'{c}')
		if not os.path.isdir(path):
			os.mkdir(path)
		vutils.save_image(
			vutils.make_grid(torch.tanh(ims), nrow=self.p.num_ims, padding=2, normalize=True)
		, os.path.join(path, f'{step}.png'))

	def save(self):
		path = os.path.join(self.p.log_dir, 'checkpoints')
		if not os.path.isdir(path):
			os.mkdir(path)
		file_name = os.path.join(path, 'data.pt')
		torch.save(torch.tanh(self.ims.cpu()), file_name)

		file_name = os.path.join(path, 'labels.pt')
		torch.save(self.labels.cpu(), file_name)

	def load_stats(self, c):
		m = torch.load(f'cifar_stats/m_{c}.pt').to(self.p.device)
		e = torch.load(f'cifar_stats/e_{c}.pt').to(self.p.device)
		c = torch.load(f'cifar_stats/c_{c}.pt').to(self.p.device)
		return m, e, c

	def train_mmd(self):
		for c in range(10):
			print(f'####### Class {c+1} #######')
			ims = self.ims[10*c:(10*c)+10]
			ims = torch.nn.Parameter(ims)
			opt = torch.optim.Adam([ims], lr=self.p.lr)
			ims.requires_grad = True
			m, e, cov_X = self.load_stats(c)
			for t in range(self.p.niter):
				self.tracker.epoch_start()
				data, label = next(self.gen)
				data = data[label == c].to(self.p.device)
				data = data[torch.randperm(data.shape[0])[:ims.shape[0]]]

				opt.zero_grad()
				encX = get_activations((data+1)/2, self.model, batch_size=ims.shape[0], device=self.p.device)
				encY = get_activations((torch.tanh(ims)+1)/2, self.model, batch_size=ims.shape[0], device=self.p.device)
				loss = mix_rbf_mmd2(encX, encY, self.sigma_list)
				loss = torch.sqrt(F.relu(loss))
				loss.backward()
				opt.step()
				self.tracker.epoch_end()
				if ((t+1)%100 == 0) or (t==0):
					self.log_interpolation(t,c,ims)
					print('[{}|{}] Loss: {:.4f}'.format(t+1, self.p.niter, loss.item()), flush=True)

			ims.requires_grad = False
			self.ims[10*c:(10*c)+10] = torch.tanh(ims) 


	def train_fid(self):
		for c in range(10):
			print(f'####### Class {c+1} #######')
			ims = self.ims[10*c:(10*c)+10]
			ims = torch.nn.Parameter(ims)
			opt = torch.optim.Adam([ims], lr=self.p.lr)
			ims.requires_grad = True
			m, e, cov_X = self.load_stats(c)
			for t in range(self.p.niter):
				self.tracker.epoch_start()
				opt.zero_grad()
				loss = fid((torch.tanh(ims)+1)/2, m, e, c, batch_size=self.p.num_ims, device=self.p.device)
				loss.backward()
				opt.step()
				self.tracker.epoch_end()
				if ((t+1)%100 == 0) or (t==0):
					self.log_interpolation(t,c,ims)
					print('[{}|{}] Loss: {:.4f}'.format(t+1, self.p.niter, loss.item()), flush=True)

			ims.requires_grad = False
			self.ims[10*c:(10*c)+10] = torch.tanh(ims) 


		self.tracker.stop()
		self.save()

	def train(self):
		if self.p.mmd:
			self.train_mmd()
		else:
			self.train_fid()
