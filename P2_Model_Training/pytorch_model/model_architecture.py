# model_architecture

import torch
import torch.nn as nn

class NIDSAutoencoder(nn.Module):
	def __init__(self, input_dim):
		super().__init__()
		
		# Encoder
		self.encoder = nn.Sequential(
			nn.Linear(input_dim, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.1),
			nn.Dropout(p = 0.2),
			
			nn.Linear(32, 16),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.1),
			
			nn.Linear(16, 8) # Latent Space
		)
		
		# Decoder
		self.decoder = nn.Sequential(
			nn.Linear(8, 16),
			nn.BatchNorm1d(16),
			nn.LeakyReLU(0.1),
			
			nn.Linear(16, 32),
			nn.BatchNorm1d(32),
			nn.LeakyReLU(0.1),
			
			nn.Linear(32, input_dim) # Linear output (critical)
		)
		
	def forward(self, x):
		z = self.encoder(x)
		x_hat = self.decoder(z)
		return x_hat
			
