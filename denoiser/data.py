# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# author: adefossez and adiyoss

import json
import logging
import os
import re

from .audio import Audioset

logger = logging.getLogger(__name__)

def match_json_files(clean_json_path, noisy_json_path):
    """匹配 clean.json 和 noisy.json 中的文件."""
    with open(clean_json_path, 'r') as f:
        clean_data = json.load(f)
    with open(noisy_json_path, 'r') as f:
        noisy_data = json.load(f)

    clean_dict = {}
    for path, size in clean_data:
        match = re.search(r'(.+)_clean.wav$', os.path.basename(path)) #change back to _clean\d+\.wav$ when needed
        if match:
            clean_dict[match.group(1)] = (path, size)

    matched_clean = []
    matched_noisy = []
    for path, size in noisy_data:
        match = re.search(r'(.+)_noisy1.wav$', os.path.basename(path)) #change back to _noisy\d+\.wav$ when needed
        if match:
            key = match.group(1)
            if key in clean_dict:
                matched_clean.append(clean_dict[key])
                matched_noisy.append((path, size))

    logger.info(f"Matched {len(matched_clean)} files.")
    return matched_clean, matched_noisy

class NoisyCleanSet:
    def __init__(self, json_dir, matching="sort", length=None, stride=None,
                 pad=True, sample_rate=None):
        """初始化 NoisyCleanSet."""
        noisy_json = os.path.join(json_dir, 'noisy.json')
        clean_json = os.path.join(json_dir, 'clean.json')

        # 使用 match_json_files 匹配 JSON 文件
        clean, noisy = match_json_files(clean_json, noisy_json)

        kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
        self.clean_set = Audioset(clean, **kw)
        self.noisy_set = Audioset(noisy, **kw)

        assert len(self.clean_set) == len(self.noisy_set)

    def __getitem__(self, index):
        return self.noisy_set[index], self.clean_set[index]

    def __len__(self):
        return len(self.noisy_set)


# When match files with more name and number complication, use this one! match_dns + match_files + NoisyCleanSet
# def match_dns(noisy, clean):  # match dns 有必要吗？- 如果已经match好了就可省略该函数。DNS = Deep Noise Supression
#     """match_dns.
#     Match noisy and clean DNS dataset filenames.

#     :param noisy: list of the noisy filenames
#     :param clean: list of the clean filenames
#     """
#     logger.debug("Matching noisy and clean for dns dataset")
#     noisydict = {}
#     extra_noisy = []
#     for path, size in noisy:
#         match = re.search(r'fileid_(\d+)\.wav$', path)
#         if match is None:
#             # maybe we are mixing some other dataset in
#             extra_noisy.append((path, size))
#         else:
#             noisydict[match.group(1)] = (path, size)
#     noisy[:] = []
#     extra_clean = []
#     copied = list(clean)
#     clean[:] = []
#     for path, size in copied:
#         match = re.search(r'fileid_(\d+)\.wav$', path)
#         if match is None:
#             extra_clean.append((path, size))
#         else:
#             noisy.append(noisydict[match.group(1)])
#             clean.append((path, size))
#     extra_noisy.sort()
#     extra_clean.sort()
#     clean += extra_clean
#     noisy += extra_noisy

# def match_files(noisy, clean, matching="sort"):
#     """match_files.
#     Sort files to match noisy and clean filenames.
#     :param noisy: list of the noisy filenames
#     :param clean: list of the clean filenames
#     :param matching: the matching function, at this point only sort is supported
#     """
#     if matching == "dns":
#         # dns dataset filenames don't match when sorted, we have to manually match them
#         match_dns(noisy, clean)
#     elif matching == "sort":
#         noisy.sort()
#         clean.sort()
#     else:
#         raise ValueError(f"Invalid value for matching {matching}")

# class NoisyCleanSet:
#     def __init__(self, json_dir, matching="sort", length=None, stride=None,
#                  pad=True, sample_rate=None):
#         """__init__.

#         :param json_dir: directory containing both clean.json and noisy.json
#         :param matching: matching function for the files
#         :param length: maximum sequence length
#         :param stride: the stride used for splitting audio sequences
#         :param pad: pad the end of the sequence with zeros
#         :param sample_rate: the signals sampling rate
#         """
#         noisy_json = os.path.join(json_dir, 'noisy.json')
#         clean_json = os.path.join(json_dir, 'clean.json')
#         with open(noisy_json, 'r') as f:
#             noisy = json.load(f)
#         with open(clean_json, 'r') as f:
#             clean = json.load(f)

#         match_files(noisy, clean, matching)
#         kw = {'length': length, 'stride': stride, 'pad': pad, 'sample_rate': sample_rate}
#         self.clean_set = Audioset(clean, **kw)
#         self.noisy_set = Audioset(noisy, **kw)

#         assert len(self.clean_set) == len(self.noisy_set)

#     def __getitem__(self, index):
#         return self.noisy_set[index], self.clean_set[index]

#     def __len__(self):
#         return len(self.noisy_set)


