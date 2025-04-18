import base64
from PIL import Image
from .files import make_dir_safe
from threading import Semaphore

def repair_webp(input_path, output_path):
    try:
        with Image.open(input_path) as img:
            # 强制重新编码为 WEBP
            img.save(output_path, "WEBP",
                     lossless=False,  # 有损压缩更兼容
                     quality=80,      # 质量参数
                     method=6)        # 编码方法 0-6
        return True
    except Exception as e:
        print(f"转换失败: {e}")
        return False

def check_webp_header(file_path):
    with open(file_path, "rb") as f:
        header = f.read(12)
        riff = header[:4] == b'RIFF'
        webp = header[8:12] == b'WEBP'
        return riff and webp

sema = Semaphore(1)

def image_to_base64(image_path):
    if not check_webp_header(image_path):
        sema.acquire()

        make_dir_safe('./temp')
        new_path = './temp/repaired.png'
        repair_webp(image_path, new_path)
        with open(new_path, "rb") as img_file:
            content = img_file.read()
        
        sema.release()
    else:
        with open(image_path, "rb") as img_file:
            content = img_file.read()
        
    encoded_bytes = base64.b64encode(content)
    encoded_str = encoded_bytes.decode("utf-8")
    return encoded_str