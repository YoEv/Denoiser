# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

from collections import namedtuple
import json
from pathlib import Path
import math
import os
import sys
import re

import torch
import torchaudio
from torch.nn import functional as F

from .dsp import convert_audio #if needed to form json file, # .dsp

Info = namedtuple("Info", ["length", "sample_rate", "channels"]) 

def get_info(path): 
    info = torchaudio.info(path)
    if hasattr(info, 'num_frames') and info.num_frames == 0: #check if info has 'num_frames'
        info.num_frames = -1
    if hasattr(info, 'num_frames'):
        return Info(info.num_frames, info.sample_rate, info.num_channels)
    else:
        siginfo = info[0]
        return Info(siginfo.length // siginfo.channels, siginfo.rate, siginfo.channels) ##

def natural_key(string):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', string)]

def find_audio_files(path, exts=[".wav"], progress=True):
    audio_files = []
    for root, _, files in os.walk(path, followlinks=True):
        for file in files:
            file = Path(root) / file
            if file.suffix.lower() in exts:
                audio_files.append(str(file.resolve()))

    audio_files.sort(key=natural_key)

    meta = []
    for idx, file in enumerate(audio_files):
        info = get_info(file)
        meta.append((file, info.length))
        if progress:
            print(format((1 + idx) / len(audio_files), " 3.1%"), end='\r', file=sys.stderr)
    return meta


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=16000,
                 channels=None, convert=True):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.convert = convert
        for _, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - self.length) / self.stride) + 1)
            else:
                examples = (file_length - self.length) // self.stride + 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index
                num_frames = self.length
                if num_frames == 0:
                    num_frames = -1
            try:
                out, sr = torchaudio.load(str(file), frame_offset=offset,num_frames=num_frames or -1)
                target_sr = self.sample_rate or sr
                target_channels = self.channels or out.shape[0]
                if self.convert:
                    out = convert_audio(out, sr, target_sr, target_channels)
                else:
                    if sr != target_sr:
                        raise RuntimeError(f"Expected {file} to have sample rate of "
                                            f"{target_sr}, but got {sr}")
                    if out.shape[0] != target_channels:
                        raise RuntimeError(f"Expected {file} to have sample rate of "
                                            f"{target_channels}, but got {sr}")
                if num_frames:
                    out = F.pad(out, (0, num_frames - out.shape[-1]))
                if self.with_path:
                    return out, file
                else:
                    return out
            except Exception as e:
                print(f"Error processing file {file}: {e}")
                return torch.zeros((1,1)) #adjust shape


if __name__ == "__main__":
    
    paths = [
        ("/Volumes/Castile/HackerProj/Denoiser/train/clean", "/Volumes/Castile/HackerProj/Denoiser/egs/debug/tr/clean.json"),
        ("/Volumes/Castile/HackerProj/Denoiser/train/noisy", "/Volumes/Castile/HackerProj/Denoiser/egs/debug/tr/noisy.json")
    ]
    
    for audio_path, output_file in paths:
        meta = find_audio_files(audio_path)
        
        with open(output_file, "w") as f:
            json.dump(meta, f, indent=4)
