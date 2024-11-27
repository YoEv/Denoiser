#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# authors: adiyoss and adefossez

import logging
import os
import json

import hydra

from denoiser.executor import start_ddp_workers

logger = logging.getLogger(__name__)


def run(args):  # 参数为args
    import torch 

    from denoiser import distrib  # 分GPU训练-DDP
    from denoiser.data import NoisyCleanSet  
    from denoiser.ConvTasNet import ConvTasNet
    from denoiser.solver import Solver
   

    # torch also initialize cuda seed if available
    torch.manual_seed(2036)  # 看要不要用Cuda吧

    import os

    # 训练数据集目录
    train_json_dir = '/Volumes/Castile/HackerProj/Denoiser/egs/debug/tr'

    # 初始化 NoisyCleanSet，自动匹配 JSON 文件
    tr_dataset = NoisyCleanSet(train_json_dir, length=48000, stride=48000, pad=True, sample_rate=16000)

    # 检查匹配结果
    print(f"Total samples: {len(tr_dataset)}")

    clean_folder = '/Volumes/Castile/HackerProj/Denoiser/train/clean'
    noise_folder = '/Volumes/Castile/HackerProj/Denoiser/train/noisy'


    clean_files = [os.path.join(clean_folder, file) for file in os.listdir(clean_folder) if file.endswith('.wav')]
    noise_files = [os.path.join(noise_folder, file) for file in os.listdir(noise_folder) if file.endswith('.wav')]


    model = ConvTasNet(sources={'clean': clean_files, 'noisy': noise_files}, N=8, L=1, B=16, H=32, P=1, X=16, R=8, audio_channels=1, norm_type="gLN", causal=False, mask_nonlinear='relu', sample_rate=16000, segment_length=44100 * 2 * 4, frame_length=400, frame_step=100) # 创建ConvTasNet模型实例，并使用args参数初始化模型
    # 开始调用的同时，就开始RUN了，所以这一步是在RUN，之后的是模型使用条件

    if args.show: # 这个args.show是出现在什么地方的？？？？？
        logger.info(model) 
        mb = sum(p.numel() for p in model.parameters()) * 4 / 2**20
        logger.info('Size: %.1f MB', mb)
        if hasattr(model, 'valid_length'):
            field = model.valid_length(1)   # 如果模型有valid——length方法，计算模型有效长度
            logger.info('Field: %.1f ms', field / args.sample_rate * 1000)
        return

    assert args.batch_size % distrib.world_size == 0  # 确定参数的batch_size批量大小，是分布式环境distrib的整数倍
    args.batch_size //= distrib.world_size

    length = int(args.segment * args.sample_rate)  # 计算一个音频的长度，对于Transformer来说很重要！！！
    stride = int(args.stride * args.sample_rate)  # 计算音频片段之间的跨度
    
    # Demucs requires a specific number of samples to avoid 0 padding during training
    if hasattr(model, 'valid_length'): # 检查模型是否有valid_length
        length = model.valid_length(length)  # 如果有valid_length,根据模型的有效长度valid_length调整音频片段的长度，
                                             # transformer同样需要！！！
    kwargs = {"matching": args.dset.matching, "sample_rate": args.sample_rate}

    ########################################################################################################
    # Building datasets and loaders 从Data里面给定数据集来训练。接口1 ！！！
    tr_dataset = NoisyCleanSet(
        args.dset.train, length=length, stride=stride, pad=args.pad, **kwargs)
    tr_loader = distrib.loader(
        tr_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    if args.dset.valid:    # 验证集
        cv_dataset = NoisyCleanSet(args.dset.valid, **kwargs)
        cv_loader = distrib.loader(cv_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        cv_loader = None
    if args.dset.test:     # 测试集
        tt_dataset = NoisyCleanSet(args.dset.test, **kwargs)
        tt_loader = distrib.loader(tt_dataset, batch_size=1, num_workers=args.num_workers)
    else:
        tt_loader = None
    data = {"tr_loader": tr_loader, "cv_loader": cv_loader, "tt_loader": tt_loader}

    if torch.cuda.is_available():
        model.cuda()      # 如果cuda可用

    # optimizer
    if args.optim == "adam":   
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    else:
        logger.fatal('Invalid optimizer %s', args.optim)
        os._exit(1)

    # Construct Solver
    solver = Solver(data, model, optimizer, args)
    solver.train()


def _main(args):
    global __file__
    # Updating paths in config
    for key, value in args.dset.items():
        if isinstance(value, str) and key not in ["matching"]:
            args.dset[key] = hydra.utils.to_absolute_path(value)
    __file__ = hydra.utils.to_absolute_path(__file__)
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        logging.getLogger("denoise").setLevel(logging.DEBUG)

    logger.info("For logs, checkpoints and samples check %s", os.getcwd())
    logger.debug(args)
    if args.ddp and args.rank is None:
        start_ddp_workers(args)
    else:
        run(args)


@hydra.main(version_base=None,config_path="conf",config_name="config")
def main(args):
    try:
        _main(args)
    except Exception:
        logger.exception("Some error happened")
        # Hydra intercepts exit code, fixed in beta but I could not get the beta to work
        os._exit(1)

from dataclasses import dataclass, field
from typing import List

@dataclass
class JobConfig:
    kv_sep: str = '='
    item_sep: str = ','
    exclude_keys: List[str] = field(default_factory=list)  # 设置 default_factory 来规避 mutable default 错误

# 在其他地方使用 JobConfig 类



if __name__ == "__main__":
    main()
