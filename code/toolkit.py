import os
import time
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