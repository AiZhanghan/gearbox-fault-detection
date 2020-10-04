import os
import time
import shutil
from functools import wraps


def timer(func):
    """计时器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        res = func(*args, **kwargs)
        end = time.time()
        print("Cost %.2f seconds" % (end - start))
        return res
    return wrapper


def get_filepath_shotname_extension(filename):
    """获取文件路径、文件名、后缀名

    Args:
        filename: str
    
    Return:
        filepath: str
        shotname: str
        extension: str
    """
    filepath, tempfilename = os.path.split(filename)
    shotname, extension = os.path.splitext(tempfilename)
    return filepath, shotname, extension


def print_shape(**kwargs):
    """
    Args:
        kwargs: dict
    """
    for key, value in kwargs.items():
        print("%s.shape: %s" % (key, value.shape))


def copy_file(source_file, target_file):
    """
    文件复制

    Args:
        source_file: str
        target_file: str
    """
    if not os.path.isfile(source_file):
        print ("%s not exist!"%(source_file))
    else:
        # 获取文件路径
        file_path = os.path.dirname(target_file)
        # 没有就创建路径 
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        # 复制文件到默认路径
        shutil.copyfile(source_file, target_file)
        print("copy %s -> %s"%(source_file, target_file))
