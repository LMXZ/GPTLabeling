from .std import *

def make_dir_safe(dir):
    if os.path.exists(dir):
        return
    os.system(f"mkdir {dir}")