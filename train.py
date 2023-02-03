import torch
import torchvision.utils as vutils
from carbontracker.tracker import CarbonTracker

import os
from fid import fid

class Trainer():
	def __init__(self, params):
		self.p = params
		if not os.path.isdir(self.p.log_dir):
			os.mkdir(self.p.log_dir)

		if self.p.init_ims:
			self.ims = torch.load('means.pt')
			self.ims = self.ims.to(self.p.device)
		else:
			self.ims = torch.randn(10*self.p.num_ims,3,32,32).to(self.p.device)

		self.labels = torch.arange(10).repeat(self.p.num_ims,1).T.flatten()

		if not os.path.isdir('./cdc_carbon'):
			os.mkdir('./cdc_carbon')
		self.tracker = CarbonTracker(epochs=self.p.niter, log_dir='./cdc_carbon/')

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
		real_m = torch.from_numpy(torch.load(f'real_m_{c}.pt')).to(self.p.device)
		real_s = torch.from_numpy(torch.load(f'real_s_{c}.pt')).to(self.p.device)
		return real_m, real_s

	def train(self):
		for c in range(10):
			print(f'####### Class {c+1} #######')
			ims = self.ims[10*c:(10*c)+10]
			ims = torch.nn.Parameter(ims)
			opt = torch.optim.Adam([ims], lr=self.p.lr)
			ims.requires_grad = True
			real_m, real_s = self.load_stats(c)
			for t in range(self.p.niter):
				self.tracker.epoch_start()
				opt.zero_grad()
				loss = fid((torch.tanh(ims)+1)/2, real_m, real_s, batch_size=self.p.num_ims, device=self.p.device)
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
