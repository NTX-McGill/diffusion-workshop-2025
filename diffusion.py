import torch
from torch import nn
import torch.nn.functional as F
import einx
import lightning
import dataclasses
from attention import SelfAttention
from attention import CrossAttention

@dataclasses.dataclass(kw_only=True)
class UnetConfig:
    in_channels:int
    n_downsample:int
    init_dim:int
    final_dim:int
    max_dim:int = None
    dims:list[int] = None
    n_heads:int = 4
    n_group:int = 32
    kernel_size:int=3
    time_dim:int=320
    d_context:int=768

class TimeEmbedding(lightning.LightningModule):
    def __init__(self, n_embd):
        super().__init__()
        self.linear_1 = nn.Linear(n_embd, 4 * n_embd)
        self.linear_2 = nn.Linear(4 * n_embd, 4 * n_embd)

    def forward(self, x):
        # x: (1, 320)

        # (1, 320) -> (1, 1280)
        x = self.linear_1(x)
        
        # (1, 1280) -> (1, 1280)
        x = F.silu(x) 
        
        # (1, 1280) -> (1, 1280)
        x = self.linear_2(x)

        return x

class UNET_ResidualBlock(lightning.LightningModule):
    def __init__(self, in_channels, out_channels, n_groups=32, n_time=1280):
        super().__init__()
        self.groupnorm_feature = nn.GroupNorm(n_groups, in_channels)
        self.conv_feature = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.linear_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(n_groups, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, feature, time):
        # feature: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)

        residue = feature
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = self.groupnorm_feature(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, In_Channels, Height, Width)
        feature = F.silu(feature)
        
        # (Batch_Size, In_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        feature = self.conv_feature(feature)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)

        # (1, 1280) -> (1, Out_Channels)
        time = self.linear_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = feature + time.unsqueeze(-1).unsqueeze(-1)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.groupnorm_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = F.silu(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(lightning.LightningModule):
    def __init__(self, n_head: int, n_embd: int,n_groups=32, d_context=768):
        super().__init__()
        channels = n_head * n_embd
        
        self.groupnorm = nn.GroupNorm(n_groups, channels, eps=1e-6)
        self.conv_input = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, in_proj_bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, d_context, in_proj_bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.linear_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.linear_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_output = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, Features, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)

        residue_long = x

        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.groupnorm(x)
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        x = self.conv_input(x)
        
        n, c, h, w = x.shape
        
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * Width)
        x = x.view((n, c, h * w))
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Height * Width, Features)
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection

        # (Batch_Size, Height * Width, Features)
        residue_short = x
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_1(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_1(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + Cross-Attention with skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_2(x)
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.attention_2(x, context)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features)
        residue_short = x

        # Normalization + FFN with GeGLU and skip connection
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x = self.layernorm_3(x)
        
        # GeGLU as implemented in the original code: https://github.com/CompVis/stable-diffusion/blob/21f890f9da3cfbeaba8e2ac3c425ee9e998d5229/ldm/modules/attention.py#L37C10-L37C10
        # (Batch_Size, Height * Width, Features) -> two tensors of shape (Batch_Size, Height * Width, Features * 4)
        x, gate = self.linear_geglu_1(x).chunk(2, dim=-1) 
        
        # Element-wise product: (Batch_Size, Height * Width, Features * 4) * (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features * 4)
        x = x * F.gelu(gate)
        
        # (Batch_Size, Height * Width, Features * 4) -> (Batch_Size, Height * Width, Features)
        x = self.linear_geglu_2(x)
        
        # (Batch_Size, Height * Width, Features) + (Batch_Size, Height * Width, Features) -> (Batch_Size, Height * Width, Features)
        x += residue_short
        
        # (Batch_Size, Height * Width, Features) -> (Batch_Size, Features, Height * Width)
        x = x.transpose(-1, -2)
        
        # (Batch_Size, Features, Height * Width) -> (Batch_Size, Features, Height, Width)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and output of the block
        # (Batch_Size, Features, Height, Width) + (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height, Width)
        return self.conv_output(x) + residue_long

class Upsample(lightning.LightningModule):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, Features, Height, Width) -> (Batch_Size, Features, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, x, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(x, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(x, time)
            else:
                x = layer(x)
        return x
    
class UNet(lightning.LightningModule):

    def __init__(self,config:UnetConfig):

        super().__init__()

        # Generate encoder dimensions dynamically
        self.encoder_dims = [config.init_dim * (2**i) for i in range(config.n_downsample)]
        if config.max_dim:
            self.encoder_dims = [min(dim, config.max_dim) for dim in self.encoder_dims]

        self.encoders = nn.ModuleList()

        current_dim = config.in_channels

        dim = self.encoder_dims[0]

        self.encoders.append(
                SwitchSequential(
                    nn.Conv2d(current_dim, dim, kernel_size=config.kernel_size, stride=1, padding=config.kernel_size // 2)
                )
            )
        
        self.encoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads, 
                         dim // config.n_heads, 
                         n_groups=config.n_group,
                         d_context=config.d_context)
                )
            )
        self.encoders.append(
            SwitchSequential(
                UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
                UNET_AttentionBlock(config.n_heads, 
                        dim // config.n_heads,
                        n_groups=config.n_group,
                        d_context=config.d_context)
            )
        )

        current_dim = dim

        for dim in self.encoder_dims[1:]:

            self.encoders.append(
                SwitchSequential(
                    nn.Conv2d(current_dim, current_dim, kernel_size=config.kernel_size, stride=2, padding=config.kernel_size // 2)
                )
            )

            self.encoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(current_dim, dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads, 
                         dim // config.n_heads,
                         n_groups=config.n_group,
                         d_context=config.d_context)
                )
            )

            self.encoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads, 
                         dim // config.n_heads,
                         n_groups=config.n_group,
                         d_context=config.d_context)
                )
            )
            
            current_dim = dim
            
        self.encoders.append(
                SwitchSequential(
                    nn.Conv2d(current_dim, current_dim, kernel_size=config.kernel_size, stride=2, padding=config.kernel_size // 2)
                )
            )

        self.encoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
                )
            )
        
        self.encoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
                )
            )

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
            UNET_AttentionBlock(config.n_heads,
                       dim // config.n_heads,
                       n_groups=config.n_group,
                       d_context=config.d_context),
            UNET_ResidualBlock(dim, dim,n_groups=config.n_group),
        )

        self.decoders = nn.ModuleList([
            SwitchSequential(UNET_ResidualBlock(dim * 2, dim,n_groups=config.n_group)),
            SwitchSequential(UNET_ResidualBlock(dim * 2, dim,n_groups=config.n_group)),
            SwitchSequential(UNET_ResidualBlock(2*dim, dim,n_groups=config.n_group),Upsample(dim)),
            SwitchSequential(UNET_ResidualBlock(dim * 2, dim,n_groups=config.n_group),
                UNET_AttentionBlock(config.n_heads,
                        dim // config.n_heads,
                        n_groups=config.n_group,
                        d_context=config.d_context)),
            SwitchSequential(UNET_ResidualBlock(dim * 2, dim,n_groups=config.n_group),
                UNET_AttentionBlock(config.n_heads,
                        dim // config.n_heads,
                        n_groups=config.n_group,
                        d_context=config.d_context)),
            
        ])

        for i in range(len(self.encoder_dims)-1):

            in_dim = self.encoder_dims[len(self.encoder_dims) - (i+1)]
            out_dim = self.encoder_dims[len(self.encoder_dims) - (i+2)]

            self.decoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(in_dim+out_dim,in_dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads,in_dim // config.n_heads,
                         n_groups=config.n_group,
                         d_context=config.d_context),
                    Upsample(in_dim)
                )
            )

            self.decoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(in_dim + out_dim,out_dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads,
                        out_dim//config.n_heads,
                        n_groups=config.n_group,
                        d_context=config.d_context),
                )
            )

            self.decoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(in_dim,out_dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads,
                        out_dim//config.n_heads,
                        n_groups=config.n_group,
                        d_context=config.d_context),
                )
            )

        self.decoders.append(
                SwitchSequential(
                    UNET_ResidualBlock(in_dim,out_dim,n_groups=config.n_group),
                    UNET_AttentionBlock(config.n_heads,
                        out_dim//config.n_heads,
                        n_groups=config.n_group,
                        d_context=config.d_context),
                )
            )


    def forward(self, x, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            x = layers(x, context, time)
            skip_connections.append(x)

        x = self.bottleneck(x, context, time)

        for layers in self.decoders:
               
            skip_connection = skip_connections.pop()
            # Since we always concat with the skip connection of the encoder, the number of features increases before being sent to the decoder's layer
            if x.shape[2:] != skip_connection.shape[2:]:
               x = F.interpolate(x,skip_connection.shape[2:])
            x = torch.cat((x, skip_connection), dim=1) 
            x = layers(x, context, time)

        return x
     
class UNET_OutputLayer(lightning.LightningModule):
    def __init__(self,
                in_channels,
                out_channels,
                n_groups=32):
        super().__init__()
        self.groupnorm = nn.GroupNorm(n_groups, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)

        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
        x = F.silu(x)
        
        # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 4, Height / 8, Width / 8)
        x = self.conv(x)
        
        # (Batch_Size, 4, Height / 8, Width / 8) 
        return x
     
class Diffusion(lightning.LightningModule):
    def __init__(self,
                 config:UnetConfig):
        super().__init__()
        self.time_embedding = TimeEmbedding(config.time_dim)
        self.unet = UNet(config)
        self.final = UNET_OutputLayer(config.init_dim, config.in_channels,config.n_group)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        # (Batch, 4, Height / 8, Width / 8) -> (Batch, 320, Height / 8, Width / 8)
        output = self.unet(latent, context, time)
        
        # (Batch, 320, Height / 8, Width / 8) -> (Batch, 4, Height / 8, Width / 8)
        output = self.final(output)
        
        # (Batch, 4, Height / 8, Width / 8)
        return output

STABLE_DIFFUSION_1P5 = UnetConfig(
    in_channels=4,            # Input channels
    n_downsample=3,           # Number of downsampling steps
    init_dim=320,             # Initial feature dimension
    final_dim=1280,           # Final bottleneck dimension
    max_dim=1280,             # Maximum feature dimension to cap at
    n_heads=8,                # Number of attention heads
    n_group=40,               # Groups for group norm in attention
    kernel_size=3,             # Kernel size for convolutions
    time_dim=320,
)