import os
import shutil

def duplicate_and_rename_files(input_dir):
    """
    复制文件并修改文件名的后缀，同时删除原始文件。
    
    Args:
        input_dir (str): 文件所在目录的路径。
    """
    # 检查输入路径是否存在
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    files = [f for f in os.listdir(input_dir) if f.endswith("_clean.wav")]

    for file_name in files:
        base_name = file_name[:-10]  # 去掉 "_clean.wav"
        ext = ".wav"
        
        for i in range(1, 6):
            new_name = f"{base_name}_clean{i}{ext}"
            src_path = os.path.join(input_dir, file_name)  
            dest_path = os.path.join(input_dir, new_name)  
            
            shutil.copy(src_path, dest_path)
            print(f"Copied {src_path} to {dest_path}")

        os.remove(os.path.join(input_dir, file_name))
        print(f"Deleted original file: {file_name}")

if __name__ == "__main__":
    input_directory = "/Volumes/Castile/HackerProj/Denoiser/train/clean"
    duplicate_and_rename_files(input_directory)

