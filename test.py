
from pesq import pesq
import librosa

def calculate_pesq(ref_audio_path, degraded_audio_path, sample_rate):
    ref_audio_signal, _ = librosa.load(ref_audio_path, sr=sample_rate)
    degraded_audio_signal, _ = librosa.load(degraded_audio_path, sr=sample_rate)

    pesq_score = pesq(sample_rate, ref_audio_signal, degraded_audio_signal, 'wb')
    return pesq_score

def main():
    # 文件路径
    ref_audio_file = '/Volumes/Castile/HackerProj/denoiser/train/clean/xs_page55_15_clean.wav'  # 替换为你的参考音频文件路径
    degraded_audio_file = '/Volumes/Castile/HackerProj/denoiser/train/noisy/xs_page55_15_noisy1.wav'  # 替换为你的降质音频文件路径

    # 样本率（Sample Rate）
    sample_rate = 48000  # 替换为你的音频文件的样本率

    # 计算 PESQ
    pesq_score = calculate_pesq(ref_audio_file, degraded_audio_file, sample_rate)
    print("PESQ Score:", pesq_score)

if __name__ == "__main__":
    main()
