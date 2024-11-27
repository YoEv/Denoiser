import os
import librosa
import soundfile as sf

def convert_and_save(input_folder, output_folder, target_sample_rate):
    """
    将音频采样率转换为目标采样率，并保存到目标文件夹。
    Args:
        input_folder (str): 输入音频文件夹路径
        output_folder (str): 输出音频文件夹路径
        target_sample_rate (int): 目标采样率
    """
    # 确保输出文件夹存在
    os.makedirs(output_folder, exist_ok=True)

    # 遍历输入文件夹中的所有音频文件
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.wav'):  # 仅处理 .wav 文件
            input_path = os.path.join(input_folder, file_name)
            output_path = os.path.join(output_folder, file_name)

            # 加载音频并转换采样率
            audio_signal, _ = librosa.load(input_path, sr=target_sample_rate)

            # 保存转换后的音频
            sf.write(output_path, audio_signal, target_sample_rate)
            #print(f"Converted and saved: {output_path}")

def main():
    # 输入文件夹
    clean_folder = '/Volumes/Castile/HackerProj/Denoiser/train/clean'
    noise_folder = '/Volumes/Castile/HackerProj/Denoiser/train/noisy'

    # 输出文件夹
    clean16_folder = '/Volumes/Castile/HackerProj/Denoiser/train/clean16'
    noisy16_folder = '/Volumes/Castile/HackerProj/Denoiser/train/noisy16'

    # 目标采样率
    target_sample_rate = 16000  # 转换为 16000 Hz

    # 转换并保存 clean 和 noisy 文件夹中的音频
    print("Converting clean files...")
    convert_and_save(clean_folder, clean16_folder, target_sample_rate)

    print("Converting noisy files...")
    convert_and_save(noise_folder, noisy16_folder, target_sample_rate)

    print("All files converted successfully.")

if __name__ == "__main__":
    main()