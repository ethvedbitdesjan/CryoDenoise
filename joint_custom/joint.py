import torch
import torch.nn as nn
import torch.nn.functional as F

class Shift2d(nn.Module):
    def __init__(self, shift: tuple[int, int]):
        super().__init__()
        self.shift = shift
        vert, horz = self.shift
        y_a, y_b = abs(vert), 0  
        x_a, x_b = abs(horz), 0  
        
        if vert < 0: y_a, y_b = y_b, y_a
        if horz < 0: x_a, x_b = x_b, x_a
        
        self.pad = nn.ZeroPad2d((x_a, x_b, y_a, y_b))
        self.crop = Crop2d((x_b, x_a, y_b, y_a))
        self.shift_block = nn.Sequential(self.pad, self.crop)

    def forward(self, x):
        return self.shift_block(x)

class Crop2d(nn.Module):
    def __init__(self, crop: tuple[int, int, int, int]):
        super().__init__()
        self.crop = crop
        
    def forward(self, x):
        l, r, t, b = self.crop
        if l == r == t == b == 0:
            return x
        return x[..., t:x.shape[-2]-b, l:x.shape[-1]-r]

class ShiftConv2d(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.shift_size = (self.kernel_size[0] // 2, 0)
        shift = Shift2d(self.shift_size)
        self.pad = shift.pad
        self.crop = shift.crop

    def forward(self, x):
        x = self.pad(x)
        x = super().forward(x)
        x = self.crop(x)
        return x
    
class JointNetwork(nn.Module):
    def __init__(self, in_channels=1, blindspot=False):
        super().__init__()
        self.blindspot = blindspot
        self.Conv2d = ShiftConv2d if blindspot else nn.Conv2d
        
        # Main U-Net backbone
        self.encoder = nn.ModuleList([
            self.make_encoder_block(in_channels, 48),  # 1->48
            self.make_encoder_block(48, 48),          # 48->48
            self.make_encoder_block(48, 48),          # 48->48
            self.make_encoder_block(48, 48),          # 48->48
            self.make_encoder_block(48, 48)           # 48->48
        ])
        
        self.decoder = nn.ModuleList([
            self.make_decoder_block(96, 96),   # 96->96 (48+48)
            self.make_decoder_block(144, 96),  # 144->96 (96+48) 
            self.make_decoder_block(144, 96),  # 144->96
            self.make_decoder_block(144, 96),   # 144->96
            self.make_decoder_block(144, 96)   # 144->96
        ])
        
        # Denoising head outputs mean and variance
        self.denoise_head = self.Conv2d(96, 2, 1)  
        
        # Detection head outputs particle probability map
        self.detect_head = self.Conv2d(96, 1, 1)

        # Noise estimation network - simpler architecture
        self.noise_estimator = NoiseEstimationNetwork(in_channels)

        if blindspot:
            self.shift = Shift2d((1, 0))

    def make_encoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            self.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2)
        )
    
    def make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            self.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear')
        )

    def reparameterize(self, x, noise_est):
        """Sample from predicted Gaussian distribution with estimated noise"""
        mu = x[:, 0:1]    # Mean
        logvar = x[:, 1:2]  # Log variance
        std = torch.exp(0.5 * logvar)
        
        # Add estimated noise variance
        total_std = torch.sqrt(std**2 + noise_est**2)
        
        # Sample using reparameterization trick
        epsilon = torch.randn_like(mu)
        z = mu + epsilon * total_std
        return z

    def forward(self, x_in):
        #print("x_in", x_in.shape)
        x = torch.clone(x_in)
        # Handle blind spot rotations if enabled
        if self.blindspot:
            rotated = [torch.rot90(x, k=k, dims=(-2,-1)) for k in range(4)]
            x = torch.cat(rotated, dim=0)
        
        # Store encoder features for skip connections
        enc_features = []
        
        # Encoder path
        for enc in self.encoder:
            x = enc(x)
            enc_features.append(x)
            #print(x.shape, 'enc')
        # Decoder path with skip connections
        for i, dec in enumerate(self.decoder):
            x = torch.cat([x, enc_features[-(i+1)]], dim=1)
            x = dec(x)
            
        if self.blindspot:
            # Apply shift and handle rotations
            x = self.shift(x)
            chunks = torch.chunk(x, 4, dim=0)
            aligned = [torch.rot90(chunk, k=-k, dims=(-2,-1)) 
                      for k, chunk in enumerate(chunks)]
            x = torch.cat(aligned, dim=1)
        
        # Get denoising parameters
        denoise_stats = self.denoise_head(x)
        
        # Get noise estimate
        noise_est = self.noise_estimator(x_in)
        
        # Sample denoised image
        denoised = self.reparameterize(denoise_stats, noise_est)
        
        # Get detection map
        detect = torch.sigmoid(self.detect_head(x))
        
        return denoise_stats, detect, noise_est

class NoiseEstimationNetwork(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        # Simplified version of the noise estimation network
        self.network = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.MaxPool2d(2),
            
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(16, 1, 1)
        )
        
    def forward(self, x):
        noise_est = self.network(x)
        # Ensure positive noise value
        return F.softplus(noise_est - 4.0) + 1e-3 #numerical stability avoid subnormals