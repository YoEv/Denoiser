import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import librosa

class ConvTasNet(nn.Module):
    def __init__(self,
                 sources,
                 N=256,
                 L=20, #每个卷积核的长度，kernal_size in encoder
                 B=256,
                 H=128,
                 P=3, # kernal_size in conv block
                 X=8,
                 R=4,
                 audio_channels=2,
                 norm_type="gLN",
                 causal=False,
                 mask_nonlinear='relu',
                 samplerate=44100,
                 segment_length=44100 * 2 * 4,
                 frame_length=400,
                 frame_step=100
                ):
        """
        Args:
            sources: list of sources
            N: Number of filters in autoencoder
            L: Length of the filters (in samples)
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """    
        
        super(ConvTasNet, self).__init__()
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        
        # Encoder
        self.encoder = nn.Conv1d(1, N, kernel_size=L, stride=L//2)
        
        # Separation module
        self.separator = TemporalConvNet(N, B, H, P, X, R)
        
        # Decoder
        self.decoder = nn.ConvTranspose1d(N, 1, kernel_size=L, stride=L//2)

    def forward(self, x):
        frames = librosa.util.frame(x.cpu().numpy(), frame_length=self.frame_length, hop_length=self.frame_step)
        frames = torch.from_numpy(frames).float()

        # 计算希望的张量大小
        desired_size = (20, 1, -1)

        # 重塑 frames 张量
        reshaped_frames = frames.reshape(*desired_size)

        # 使用重塑后的 frames 进行后续处理
        encoded_frames = self.encoder(reshaped_frames)
        #encoded_frames = self.encoder(frames.view(20, 1, -1)) #old ver using .view
        separated_frames = self.separator(encoded_frames)  #############
        reconstructed_frames = self.decoder(separated_frames)
        
        return reconstructed_frames

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, kernel_size=None, dilation=1):
        if kernel_size==None:
            kernel_size=3
            
        super(TemporalConvNet, self).__init__()
        layers = []
        for _ in range(B):
            dilation = 2 ** _ if _ < R else 2 ** ((_ - R) % X)
            layers += [TemporalBlock(N, H, P, kernel_size, dilation)]
        self.network = nn.Sequential(*layers)

    def forward(self, x): ##########
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, in_channels=128, out_channels=128, kernel_size=3, dilation=1, padding=0, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm([20, 128, 34397])  # 注意：LayerNorm的参数需要和输入张量的形状相匹配
    
    def forward(self, x):
        residual = x
        x = F.relu(self.layer_norm(x))  # 问题可能出现在这里，需要进一步检查
        x = self.dropout(x)
        x = F.relu(self.layer_norm(x))  # 你也许需要检查这里
        x += residual
        return x



'''
class TemporalBlock(nn.Module):
    def __init__(self, input: Tensor, weight: Tensor, bias, in_channels=1, out_channels=1, kernel_size=3, dilation=1, padding=0, dropout=0.2):
        Tensor = [20,128,128]

        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input, weight, in_channels, out_channels, kernel_size, bias=False)
        self.conv2 = nn.Conv1d(input, weight, in_channels, out_channels, kernel_size, bias=False)
    
        #return F.conv1d(input, weight, bias, self.stride,self.padding, self.dilation, self.groups)
        #self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
         #                      padding=dilation, dilation=dilation)
        #self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size,
        #                       padding=dilation, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(Tensor)  # THIS HAS TO MATCH WITH x = F.relu's input!!!!

    def forward(self, x):
        residual = x
        Tensor=[20,128,34397]
        self.layer_norm = nn.LayerNorm(Tensor, elementwise_affine = False)  # THIS HAS TO MATCH WITH x = F.relu's input!!!!

       #Reshape x to match the expected shape for LayerNorm
       #B, L, C = x.size()  # Get batch size, length, and channels
       #x = x.transpose(1, 2).contiguous()  # Reshape for LayerNorm
       #x = x.view(B * C, L)  # Reshape to [*=B*C, L] for LayerNorm
       # x = self.layer_norm(x)
       # x = x.view(B, C, L)  # Reshape back to original shape
       # x = x.transpose(1, 2).contiguous()  # Transpose back to [B, L, C]

       # x = F.relu(self.conv1(x))        
        x = F.relu(self.layer_norm(x))  #这个地方的方程有问题。另外还要确认每一次的输入和输出是不是应该是不一样的但是对应的值.34397或者34401是卷积的结果
        x = self.dropout(x)
        x = F.relu(self.layer_norm(x))
        x += residual #####################
        return x
'''