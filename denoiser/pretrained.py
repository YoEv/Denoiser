# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez

import logging

import torch.hub

from .ConvTasNet import ConvTasNet
from .utils import deserialize_model

logger = logging.getLogger(__name__)
ROOT = "https://dl.fbaipublicfiles.com/adiyoss/denoiser/"
DNS_48_URL = ROOT + "dns48-11decc9d8e3f0998.th"
DNS_64_URL = ROOT + "dns64-a7761ff99a7d5bb6.th"
MASTER_64_URL = ROOT + "master64-8a5dfb4bb92753dd.th"
VALENTINI_NC = ROOT + 'valentini_nc-93fc4337.th'  # Non causal Demucs on Valentini
#do I have to add a .th file for my two set of checkpoint?


# def _convtasnet(pretrained, url, **kwargs):
#     model = ConvTasNet(**kwargs, sample_rate=16_000)
#     if pretrained:
#         state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
#         model.load_state_dict(state_dict)
#     return model

def _convtasnet(pretrained=False, checkpoint=None, **kwargs):
    """
    Helper function to create a ConvTasNet model, optionally load weights from checkpoint.
    """
    model = ConvTasNet(**kwargs, sample_rate=16_000)
    if pretrained:
        if not checkpoint:
            raise ValueError("Pretrained model requested, but no checkpoint provided.")
        logger.info(f"Loading weights from checkpoint: {checkpoint}")
        state_dict = torch.load(checkpoint, map_location='cpu')
        model.load_state_dict(state_dict)
    return model

def convtasnet_small(pretrained=False, checkpoint=None):
    # Small version of ConvTasNet
    return _convtasnet(pretrained, checkpoint=checkpoint, sources={"clean": None}, N=8, L=10, B=16, H=32, P=1, X=8, R=4)

def convtasnet_large(pretrained=False, checkpoint=None):
    # Large version of ConvTasNet
    return _convtasnet(pretrained, checkpoint=checkpoint, sources={"clean": None}, N=16, L=10, B=32, H=64, P=3, X=16, R=8)
# def dns48(pretrained=True):
#     return _convtasnet(pretrained, DNS_48_URL, hidden=48)


# def dns64(pretrained=True):
#     return _convtasnet(pretrained, DNS_64_URL, hidden=64)


# def master64(pretrained=True):
#     return _convtasnet(pretrained, MASTER_64_URL, hidden=64)


# def valentini_nc(pretrained=True):
#     return _convtasnet(pretrained, VALENTINI_NC, hidden=64, causal=False, stride=2, resample=2)

def add_model_flags(parser):
    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("-m", "--model_path", help="Path to local trained model.")
    group.add_argument("--convtasnet_small", action="store_true",
                       help="Use small version of ConvTasNet.")
    group.add_argument("--convtasnet_large", action="store_true",
                       help="Use large version of ConvTasNet.")


# def add_model_flags(parser):
#     group = parser.add_mutually_exclusive_group(required=False)
#     group.add_argument("-m", "--model_path", help="Path to local trained model.")
#     group.add_argument("--dns48", action="store_true",
#                        help="Use pre-trained real time H=48 model trained on DNS.")
#     group.add_argument("--dns64", action="store_true",
#                        help="Use pre-trained real time H=64 model trained on DNS.")
#     group.add_argument("--master64", action="store_true",
#                        help="Use pre-trained real time H=64 model trained on DNS and Valentini.")
#     group.add_argument("--valentini_nc", action="store_true",
#                        help="Use pre-trained H=64 model trained on Valentini, non causal.")

def get_model(args):
    """
    Load local model package or torchhub pre-trained model.
    """
    if args.model_path:
        logger.info("Loading model from %s", args.model_path)
        pkg = torch.load(args.model_path, 'cpu')
        if 'model' in pkg:
            if 'best_state' in pkg:
                pkg['model']['state'] = pkg['best_state']
            model = deserialize_model(pkg['model'])
        else:
            model = deserialize_model(pkg)
    elif args.convtasnet_small:
        logger.info("Loading small ConvTasNet model.")
        model = convtasnet_small()
    elif args.convtasnet_large:
        logger.info("Loading large ConvTasNet model.")
        model = convtasnet_large()
    else:
        logger.info("Loading default ConvTasNet model (small).")
        model = convtasnet_small()
    logger.debug(model)
    return model
