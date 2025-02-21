import torch
from torch import nn
import torch.nn.functional as F
import einx
import lightning
from decoder import VAE_AttentionBlock,VAE_ResidualBlock,EncoderConfig

class Encoder(lightning.LightningModule):

	def __init__(self,config:EncoderConfig):

		super().__init__()

		self.layers = nn.ModuleList()

		self.layers.append(nn.Conv2d(in_channels=config.in_channels,out_channels=config.init_dim,
					  kernel_size=config.kernel_size,padding=config.kernel_size//2))
		
		self.layers.append(VAE_ResidualBlock(config.init_dim,config.init_dim,config.n_group))
		self.layers.append(VAE_ResidualBlock(config.init_dim,config.init_dim,config.n_group))

		dim = config.init_dim

		if config.dims is None:
			dims = []
			d = config.init_dim
			for _ in range(config.n_downsample-1):
				dims.append(2*d)
				d = 2*d
		else:
			assert len(config.dims) == config.n_downsample
			dims = config.dims

		print(dims)

		for out_dim in dims:
			self.layers.append(nn.Conv2d(dim,dim,config.kernel_size,stride=2,padding=0))
			self.layers.append(VAE_ResidualBlock(dim,out_dim,config.n_group))
			self.layers.append(VAE_ResidualBlock(out_dim,out_dim,config.n_group))
			dim = out_dim

		self.layers.append(nn.Conv2d(dim,dim,config.kernel_size,stride=2,padding=0))
		self.layers.append(VAE_ResidualBlock(dim,dim,config.n_group))
		self.layers.append(VAE_ResidualBlock(dim,dim,config.n_group))

		self.layers.append(VAE_ResidualBlock(dim,dim,config.n_group))

		self.layers.append(VAE_AttentionBlock(dim,config.n_heads,config.n_group))

		self.layers.append(VAE_ResidualBlock(dim,dim,config.n_group))
		
		self.layers.append(nn.GroupNorm(dim,dim,config.n_group))
		self.layers.append(nn.SiLU())
		self.layers.append(nn.Conv2d(dim,config.final_dim,kernel_size=config.kernel_size,
						   padding=config.kernel_size//2))
		self.layers.append(nn.Conv2d(config.final_dim,config.final_dim,kernel_size=1,padding=0))

	def forward(self,x,noise):
		
		mean,log_var = self.stats_forward(x)
		log_var = torch.clamp(log_var,-30,20)

		var = torch.exp(log_var)

		std = torch.sqrt(var)

		x = mean + std * noise

		x *= 0.18215

		return x
	
	def stats_forward(self,x):
		for module in self.layers:
			if getattr(module,"stride",None) == (2,2):
				x = F.pad(x, (0,1,0,1))
			x = module(x)

		mean,log_var = einx.rearrange("b (k d) h w -> k b d h w",x,k=2)
		return mean,log_var

	def full_forward(self,x):

		mean,log_var = self.stats_forward(x)
		var = torch.exp(log_var)
		noise = torch.randn_like(log_var)
		z = mean + var*noise
		z *= 0.18215
		return z,mean,log_var
	
STABLE_DIFFUSION_VAE = EncoderConfig(
	in_channels=3,
	n_downsample=3,
	decoder_dims=[512,512,256,128],
	init_dim=128,
	final_dim=8,
	n_group=32,
)