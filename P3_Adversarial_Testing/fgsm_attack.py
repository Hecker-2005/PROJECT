import torch
import numpy as np

def fgsm_attack(model, x, epsilon):
	"""
	FGSM attack on reconstruction loss.
	"""
	x_adv = x.clone().detach().requires_grad_(True)
	recon = model(x_adv)
	loss = torch.mean((x_adv - recon) ** 2)
	loss.backward()
	perturbation = epsilon * x_adv.grad.sign()
	return (x_adv + perturbation).detach()
	
def run_fgsm(
	model,
	X, 
	epsilon,
	validity_filter,
	device = "cpu",
	batch_size = 256
):
	model.eval()
	
	valid_samples = []
	invalid_count = 0
	
	for i in range(0, len(X), batch_size):
		x_batch = torch.tensor(
			X[i:i + batch_size],
			dtype = torch.float32,
			device = device
		)
		
		x_adv = fgsm_attack(model, x_batch, epsilon)
			
		for j in range(len(x_adv)):
			x_orig = x_batch[j].cpu().numpy()
			x_candidate = x_adv[j].cpu().numpy()
			
			x_proj = validity_filter.project(x_candidate, x_orig)
			
			if validity_filter.is_valid(x_proj):
				valid_samples.append(x_proj)
			else:
				invalid_count += 1
				
	return np.array(valid_samples), invalid_count

