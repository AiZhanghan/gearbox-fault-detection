import os
import time
from functools import wraps
from sklearn.decomposition import PCA


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


def tuning_pca(feature, n_components):
    """
    Args:
        feature: df.DataFrame
        n_components: float
    """
    pca = PCA(n_components=n_components)
    pca.fit(feature)
    print("pca.components_.shape: %s, sum(pca.explained_variance_ratio_: %.4f"
        % (pca.components_.shape, sum(pca.explained_variance_ratio_)))
        