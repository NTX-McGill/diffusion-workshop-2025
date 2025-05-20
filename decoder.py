import torch
from torch import nn
import torch.nn.functional as F
import einx
import lightning
from custom_ddpm.attention import SelfAttention
import dataclasses

@dataclasses.dataclass(kw_only=True)
class EncoderConfig:
	in_channels:int
	n_downsample:int
	init_dim:int
	final_dim:int
	decoder_dims:list[int]
	max_dim:int = None
	dims:list[int] = None
	n_heads:int = 4
	n_group:int = 32
	kernel_size:int=3


class VAE_AttentionBlock(lightning.LightningModule):
	def __init__(self, channels,n_heads=1,n_groups=32):
		super().__init__()
		self.groupnorm = nn.GroupNorm(n_groups, channels)
		self.attention = SelfAttention(n_heads, channels)
	
	def forward(self, x):
		# x: (Batch_Size, Features, Height, Width)

		residue = x 

		# (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
		x = self.groupnorm(x)

		n, c, h, w = x.shape
		
		# (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
		x = x.view((n, c, h * w))
		
		# (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features). Each pixel becomes a feature of size "Features", the sequence length is "Height * Width".
		x = x.transpose(-1, -2)
		
		# Perform self-attention WITHOUT mask
		# (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
		x = self.attention(x)
		
		# (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
		x = x.transpose(-1, -2)
		
		# (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
		x = x.view((n, c, h, w))
		
		# (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width) 
		x += residue

		# (Batch_Size, Features, Height, Width)
		return x 

class VAE_ResidualBlock(lightning.LightningModule):
	def __init__(self, 
				 in_channels, 
				 out_channels,
				 n_groups=32):
		super().__init__()
		self.groupnorm_1 = nn.GroupNorm(n_groups, in_channels)
		self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

		self.groupnorm_2 = nn.GroupNorm(n_groups, out_channels)
		self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

		if in_channels == out_channels:
			self.residual_layer = nn.Identity()
		else:
			self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
	
	def forward(self, x):
		# x: (Batch_Size, In_Channels, Height, Width)

		residue = x

		# (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
		x = self.groupnorm_1(x)
		
		# (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
		x = F.silu(x)
		
		# (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
		x = self.conv_1(x)
		
		# (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
		x = self.groupnorm_2(x)
		
		# (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
		x = F.silu(x)
		
		# (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
		x = self.conv_2(x)
		
		# (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
		return x + self.residual_layer(residue)

class Decoder(lightning.LightningModule):

	def __init__(self,config:EncoderConfig):

		super().__init__()

		assert config.n_downsample == len(config.decoder_dims)-1

		self.layers = nn.ModuleList()

		if (config.max_dim is None) & (config.dims is None):
			d_in = config.init_dim*(2**(config.n_downsample-1))
		elif (config.max_dim is None) & (config.dims is not None):
			d_in = config.dims[-1]
		else:
			d_in = config.max_dim

		self.layers.append(nn.Conv2d(config.final_dim//2,config.final_dim//2,kernel_size=1,padding=0))
		self.layers.append(nn.Conv2d(config.final_dim//2,d_in,kernel_size=3,padding=1))

		self.layers.append(VAE_ResidualBlock(d_in,d_in,config.n_group))
		self.layers.append(VAE_AttentionBlock(d_in,config.n_heads,config.n_group))

		for idx,d_out in enumerate(config.decoder_dims[:-1]):

			self.layers.append(VAE_ResidualBlock(d_in,d_out,config.n_group))
			self.layers.append(VAE_ResidualBlock(d_out,d_out,config.n_group))
			self.layers.append(VAE_ResidualBlock(d_out,d_out,config.n_group))
			if idx == 0:
				self.layers.append(VAE_ResidualBlock(d_out,d_out,config.n_group))

			self.layers.append(nn.Upsample(scale_factor=2))
			self.layers.append(nn.Conv2d(d_out,d_out,kernel_size=config.kernel_size,padding=config.kernel_size//2))

			d_in = d_out

		d_out = config.decoder_dims[-1]

		self.layers.append(VAE_ResidualBlock(d_in,d_out,config.n_group))
		self.layers.append(VAE_ResidualBlock(d_out,d_out,config.n_group))
		self.layers.append(VAE_ResidualBlock(d_out,d_out,config.n_group))

		self.layers.append(nn.GroupNorm(config.n_group,d_out))
		self.layers.append(nn.SiLU())
		self.layers.append(nn.Conv2d(d_out,config.in_channels,kernel_size=config.kernel_size,
							   padding=config.kernel_size//2))

	def forward(self,x):

		x /= 0.18215

		for module in self.layers:
			if isinstance(module,nn.Upsample):
				dtype = x.dtype
				x = x.to(torch.float32)
				x = module(x).to(dtype)
			else:
				x = module(x)
		return x