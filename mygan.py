import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.nn.utils import clip_grad_norm_

import numpy as np
import matplotlib.pyplot as plt

from time import time
import datetime


def to_numpy_image(img):
	return img.detach().cpu().view(*img.shape).transpose(0, 1).transpose(1, 2).numpy()


def weights_init(m):
	classname = m.__class__.__name__
	if classname.find('Conv') != -1:
		nn.init.normal_(m.weight.data, 0.0, 0.02)
	elif classname.find('BatchNorm') != -1:
		nn.init.normal_(m.weight.data, 1.0, 0.02)
		nn.init.constant_(m.bias.data, 0)


class GAN:
	def __init__(self, D, G, cuda=False):
		if cuda:
			self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
			
		else:
			self.device = torch.device("cpu")
		print(self.device)
		# model
		self.G = G.to(self.device)
		self.D = D.to(self.device)

		#stats and logs
		self.epoch = 0
		self.training_time = 0
		self.G_losses = []
		self.D_losses = []

		# hyperparameters

		D_learning_rate = 1e-4
		G_learning_rate = 1e-4

		# loss
		self.criterion = nn.BCELoss()


		self.latent_size = next(self.G.parameters()).size()[0]
		self.optim_G = torch.optim.Adam(self.G.parameters(), lr = D_learning_rate)
		self.optim_D = torch.optim.Adam(self.D.parameters(), lr = G_learning_rate)

		self.G.apply(weights_init)
		self.D.apply(weights_init)


	def download_params(self, model_path):

		checkpoint = torch.load(model_path)

		self.epoch = checkpoint['epoch']
		self.training_time = checkpoint['time']

		self.G_losses = checkpoint['G_losses']
		self.D_losses = checkpoint['D_losses']

		self.G.load_state_dict(checkpoint['G_state_dict'])
		self.optim_G.load_state_dict(checkpoint['G_optimizer_state_dict'])

		self.D.load_state_dict(checkpoint['D_state_dict'])
		self.optim_D.load_state_dict(checkpoint['D_optimizer_state_dict'])

		print(str(datetime.timedelta(self.training_time)))


	def save_params(self, path):
		self.training_time += time() - self.start_time
		self.start_time = time()

		torch.save({
			'time': self.training_time,
			'epoch': self.epoch,
			'G_state_dict': self.G.state_dict(),
			'D_state_dict': self.D.state_dict(),
			'G_optimizer_state_dict': self.optim_G.state_dict(),
			'D_optimizer_state_dict': self.optim_D.state_dict(),
			'G_losses': self.G_losses,
			'D_losses': self.D_losses
			}, path)

		print('Succesfully saved!')
		print(str(datetime.timedelta(self.training_time)))


	def generate_latent(self, n = 1):
		#n - number of samples
		return torch.randn(n, self.latent_size, 1, 1).to(self.device)


	def sample_and_plot(self, n = 1):
		side = int(n**0.5)
		z = self.generate_latent(n)
		generated = self.G(z)
		
		f, axarr = plt.subplots(side, side, figsize=(10,10))
		for i in range(n):
			img = to_numpy_image(generated[i]) * 255
			axarr[i//side,i%side].imshow(img.astype(np.uint8))
			axarr[i//side,i%side].axis('off')  
			plt.pause(0.1)  
		f.show()


	def add_dataset(self, dataset, batch_size):
		self.loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


	def train(self, saving=False, D_speed=5, G_speed=1):
		self.start_time = time()
		self.sample_and_plot(4)


		while True:
			for iters, (data, _) in enumerate(self.loader, 1): # TODO

				batch_size = len(data)
				y_zeros = torch.zeros(batch_size).to(self.device)
				y_ones = (torch.ones(batch_size)*0.93).to(self.device)
				data += torch.randn(data.size())/10
				data = data.to(self.device)


				#1) D ascent: max log(D(x)) + log(1 - D(G(z)))
				self.D.zero_grad()
				#a)
				D_output = self.D(data)
				D_loss = self.criterion(torch.flatten(D_output), y_ones)
				D_loss.backward(retain_graph=True)
				#b)
				z = self.generate_latent(batch_size) 
				G_output = self.G(z).detach()
				D_loss = self.criterion(torch.flatten(D_output), y_zeros) 

				clip_grad_norm_(self.D.parameters(), 0.5)
				if iters % D_speed == 0:
					self.optim_D.step()


				#2) G ascent: max log(D(G(z)))
				self.G.zero_grad()
				z = self.generate_latent(batch_size)
				G_output = self.G(z)
				D_output = self.D(G_output)
				G_loss = self.criterion(torch.flatten(D_output), y_ones) 
				G_loss.backward(retain_graph=True)

				clip_grad_norm_(self.G.parameters(), 0.5)
				if iters % G_speed == 0:
					self.optim_G.step()


				#logging
				self.D_losses.append(D_loss.item())
				self.G_losses.append(G_loss.item())
			
				if iters % 100 == 0:
					
					print(f'{iters}/{len(loader)}')
					print(f'  G loss: {G_loss}')
					print(f'  D loss: {D_loss}')
					plt.figure(figsize=(10,5))
					plt.plot(self.G_losses, label="G", color = 'orange')
					plt.plot(self.D_losses, label="D", color = 'blue')

					plt.xlabel("Iterations")
					plt.ylabel("Loss")
					plt.legend()
					plt.show()

					self.sample_and_plot(16)

				if iters % 200 == 0:
					print(str(datetime.timedelta(self.training_time)))
					self.save_params(self.path)

			self.epoch += 1