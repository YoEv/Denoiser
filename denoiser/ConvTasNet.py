import torch
# from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
# import librosa
import inspect 
# import math
import time

class ConvTasNet(nn.Module):
    def __init__(self,
                 sources,
                 N=8, # # of filters in autoencorder, out_channel(encoder),in_channel(decoder)
                 L=1, #10 or 20 kernal_size in encoder，但在实际训练中使用的是1
                 B=16, #最细的部分，所以B>N其实是扩张了。
                 H=32,
                 P=3, #kernal_size in conv block
                 X=16,
                 R=8,
                 audio_channels=1,
                #  norm_type="gLN", #to save compute sources this time
                #  causal=False,
                #  mask_nonlinear='relu',
                 sample_rate=16000,
                #  segment_length=44100 * 2 * 4, #处理片段长度
                 frame_length=400,
                 frame_step=100,
                ):
        """
        Args:
            sources: list of sources
            N: Number of filters in autoencoder
            L: Length of the filters (in samples) - kernal_size in encoder
            B: Number of channels in bottleneck 1 × 1-conv block
            H: Number of channels in convolutional blocks
            P: Kernel size in convolutional blocks
            X: Number of convolutional blocks in each repeat
            R: Number of repeats
            norm_type: BN, gLN, cLN
            causal: causal or non-causal
            mask_nonlinear: use which non-linear function to generate mask
        """    
        args, _, _, kwargs = inspect.getargvalues(inspect.currentframe())
        self._init_args_kwargs = (args, kwargs)

        super(ConvTasNet, self).__init__()
        
        self.frame_length = frame_length
        self.frame_step = frame_step
        self.sample_rate = sample_rate
        self.audio_channels = audio_channels

        # Encoder
        self.encoder = nn.Conv1d(in_channels=1, out_channels=N, kernel_size=L, stride=1, padding=0) 
        
        # Separation module - 最核心的部分，encode之后进入Temporal ConvNet，再分散process到每一个Temporal Blocks
        self.separator = TemporalConvNet(N, B, H, P, X, R) 

        # Decoder，恢复空间维度
        self.decoder = nn.ConvTranspose1d(in_channels=N, out_channels=1, kernel_size=L, stride=1, padding=0, output_padding=0) 

    def forward(self, x):
        
        encoded_frames = self.encoder(x) 
        
        separated_frames = self.separator(encoded_frames)  

        reconstructed_frames = self.decoder(separated_frames)

        return reconstructed_frames
        

class TemporalConvNet(nn.Module):
    def __init__(self, N, B, H, P, X, R, kernel_size=None, dilation=None):

        super(TemporalConvNet, self).__init__()
        layers = []
        for _ in range(B):
            dilation = 2 ** _ if _ < R else 2 ** ((_ - R) % X)
            layers += [TemporalBlock(N, H)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class TemporalBlock(nn.Module):
    def __init__(self, N, H, P, dilation=1, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(N, H, P, padding=P//2, dilation=dilation)
        self.conv2 = nn.Conv1d(H, N, P, padding=P//2, dilation=dilation)
        self.dropout = nn.Dropout(min(dropout, 1.0)) 
        self.layer_norm1 = nn.LayerNorm(H)  # Normalize over the channel dimension
        self.layer_norm2 = nn.LayerNorm(N)  # Normalize over the channel dimension

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = F.relu(x)
        x = x.transpose(1, 2)  
        x = self.layer_norm1(x)
        x = x.transpose(1, 2)  
        x = self.dropout(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = x.transpose(1, 2)   
        x = self.layer_norm2(x)
        x = x.transpose(1, 2)
        x = self.dropout(x)
        x += residual
        return x


class ConvTasNetStreamer:
    """
    Streaming implementation for ConvTasNet. It supports being fed with any amount
    of audio at a time. You will get back as much audio as possible at that
    point.

    Args:
        - convtasnet (ConvTasNet): ConvTasNet model.
        - dry (float): amount of dry (e.g., input) signal to keep.
        - num_frames (int): number of frames to process at once.
    """
    def __init__(self, convtasnet, dry=0, num_frames=1):
        device = next(iter(convtasnet.parameters())).device
        self.convtasnet = convtasnet
        self.dry = dry
        self.num_frames = num_frames
        self.stride = 16  # Example value, replace with actual stride of ConvTasNet
        self.frame_length = 16 * num_frames  # Example calculation
        self.total_length = self.frame_length  # Adjust as needed
        self.pending = torch.zeros(convtasnet.input_channels, 0, device=device)

    def flush(self):
        """
        Flush remaining audio by padding it with zero. Call this when you have no more input.
        """
        pending_length = self.pending.shape[1]
        padding = torch.zeros(self.convtasnet.input_channels, self.total_length, device=self.pending.device)
        out = self.feed(padding)
        return out[:, :pending_length]

    def feed(self, wav):
        """
        Apply the ConvTasNet model to the given audio in a streaming fashion.
        """
        begin = time.time()

        if wav.dim() != 2:
            raise ValueError("Input wav should be two-dimensional.")
        chin, _ = wav.shape
        if chin != self.convtasnet.input_channels:
            raise ValueError(f"Expected {self.convtasnet.input_channels} channels, got {chin}")

        self.pending = torch.cat([self.pending, wav], dim=1)
        outs = []
        while self.pending.shape[1] >= self.total_length:
            frame = self.pending[:, :self.total_length]
            dry_signal = frame[:, :self.stride]
            out = self.convtasnet(frame.unsqueeze(0)).squeeze(0)  # Process the frame
            out = self.dry * dry_signal + (1 - self.dry) * out[:, :self.stride]
            outs.append(out)
            self.pending = self.pending[:, self.stride:]

        if outs:
            out = torch.cat(outs, 1)
        else:
            out = torch.zeros(chin, 0, device=wav.device)
        return out