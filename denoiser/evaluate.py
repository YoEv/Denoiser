
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import logging
import sys

from pesq import pesq
from pystoi import stoi
import torch
import numpy as np

from .data import NoisyCleanSet
from .enhance import add_flags, get_estimate
from . import distrib, pretrained
from .utils import bold, LogProgress
from . import ConvTasNet

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser(
        'denoiser.evaluate',
        description='Speech enhancement using Demucs - Evaluate model performance')
add_flags(parser)
parser.add_argument('--data_dir', help='directory including noisy.json and clean.json files')
parser.add_argument('--matching', default="sort", help='set this to dns for the dns dataset.')
parser.add_argument('--no_pesq', action="store_false", dest="pesq", default=True,
                    help="Don't compute PESQ.")
parser.add_argument('-v', '--verbose', action='store_const', const=logging.DEBUG,
                    default=logging.INFO, help="More loggging")

def evaluate(args, model=None, data_loader=None):
    total_pesq = 0
    total_stoi = 0
    total_cnt = 0
    updates = 5

    # Load model
    # if not model:
    #     model = pretrained.get_model(args).to(args.device)
    # model.eval()
    if not model:
    # 直接实例化一个 ConvTasNet 模型，传入所需的参数
        model = ConvTasNet(
            sources=args.sources,
            N=8,
            L=1,
            B=16,
            H=32,
            P=1,
            X=16,
            R=8,
            audio_channels=2,
            norm_type="gLN",
            causal=False,
            mask_nonlinear='relu',
            sample_rate=16000,
            segment_length=44100 * 2 * 4, #? May need to reconsider
            frame_length=400,
            frame_step=100,
        ).to(args.device)


    # Load data
    if data_loader is None:
        dataset = NoisyCleanSet(args.data_dir,
                                matching=args.matching, sample_rate=model.sample_rate)
        data_loader = distrib.loader(dataset, batch_size=1, num_workers=2)
    pendings = []
    with ProcessPoolExecutor(args.num_workers) as pool:
        with torch.no_grad():
            iterator = LogProgress(logger, data_loader, name="Eval estimates")
            for i, data in enumerate(iterator):
                # Get batch data
                noisy, clean = [x.to(args.device) for x in data]
                if args.device == 'cpu':
                    pendings.append(
                        pool.submit(_estimate_and_run_metrics, clean, model, noisy, args))
                else:
                    estimate = get_estimate(model, noisy, args)
                    estimate = estimate.cpu()
                    clean = clean.cpu()
                    pendings.append(
                        pool.submit(_run_metrics, clean, estimate, args, model.sample_rate))
                total_cnt += clean.shape[0]

        for pending in LogProgress(logger, pendings, updates, name="Eval metrics"):
            pesq_i, stoi_i = pending.result()
            total_pesq += pesq_i
            total_stoi += stoi_i

    metrics = [total_pesq, total_stoi]
    pesq, stoi = distrib.average([m/total_cnt for m in metrics], total_cnt)
    logger.info(bold(f'Test set performance:PESQ={pesq}, STOI={stoi}.'))
    return pesq, stoi


def _estimate_and_run_metrics(clean, model, noisy, args):
    estimate = get_estimate(model, noisy, args)
    return _run_metrics(clean, estimate, args, sr=16000)

def _run_metrics(clean, estimate, args, sr):
    estimate = estimate.numpy()[:, 0]
    clean = clean.numpy()[:, 0]
    if args.pesq:
        pesq_i = get_pesq(clean, estimate, sr=sr)
    else:
        pesq_i = 0
    stoi_i = get_stoi(clean, estimate, sr=sr)
    return pesq_i, stoi_i

def get_pesq(ref_sig, out_sig, sr):
    """Calculate PESQ.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        PESQ
    """
    pesq_val = 0
    for i in range(len(ref_sig)):
        pesq_val += pesq(sr, ref_sig[i], out_sig[i], 'wb')
    return pesq_val 

def get_stoi(ref_sig, out_sig, sr):
    """Calculate STOI.
    Args:
        ref_sig: numpy.ndarray, [B, T]
        out_sig: numpy.ndarray, [B, T]
    Returns:
        STOI
    """
    stoi_val = 0
    # for i in range(len(ref_sig)):
    #     min_length = min(len(ref_sig[i]), len(out_sig[i]))
    #     ref_sig[i] = ref_sig[i][:min_length]
    #     out_sig[i] = out_sig[i][:min_length]
    #     stoi_val += stoi(ref_sig[i], out_sig[i], sr, extended=False)
    aligned_ref = []
    aligned_out = []

    for i in range(len(ref_sig)):
        max_length = max(len(ref_sig[i]), len(out_sig[i]))
        
        # 如果长度不足，则填充；如果长度超过，则裁剪
        ref = np.pad(ref_sig[i], (0, max_length - len(ref_sig[i])), mode='constant') if len(ref_sig[i]) < max_length else ref_sig[i][:max_length]
        out = np.pad(out_sig[i], (0, max_length - len(out_sig[i])), mode='constant') if len(out_sig[i]) < max_length else out_sig[i][:max_length]

        aligned_ref.append(ref)
        aligned_out.append(out)
        # print(f"Aligned signal {i}: ref length={len(ref)}, out length={len(out)}")

    # 计算 STOI
    for ref, out in zip(aligned_ref, aligned_out):
        stoi_val += stoi(ref, out, sr, extended=False)
    return stoi_val


def main():
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, level=args.verbose)
    logger.debug(args)
    pesq, stoi = evaluate(args)
    json.dump({'pesq': pesq, 'stoi': stoi}, sys.stdout)
    sys.stdout.write('\n')


if __name__ == '__main__':
    main()
