import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt


def plot_line(anomaly_scores_train, label_train, anomaly_scores_test, 
    label_test, threshold, wind_farm, wind_turbine):
    """
    Args:
        anomaly_scores_train: pd.DataFrame
        anomaly_scores_test: pd.DataFrame
        label_train: pd.DataFrame
        label_test: pd.DataFrame
        threshold: float
        wind_farm: str
        wind_turbine: str
    """
    set_figsize(figsize=(8, 8))
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)
    
    anomaly_scores = [anomaly_scores_train, anomaly_scores_test]
    label = [label_train, label_test]
    
    for i in range(2):    
        # score
        anomaly_scores[i].plot(ax=ax[i], linewidth=2, alpha=0.5)
        # title
        ax[i].set_title("%s #%s %s" % (wind_farm, wind_turbine, 
            "train" if i == 0 else "test"))
        # threshold
        ax[i].plot(anomaly_scores[i].index, [threshold] * 
            len(anomaly_scores[i]), "r--", label="threshold")
        # label
        ax[i].scatter(x=anomaly_scores[i][label[i].label].index, 
            y=anomaly_scores[i][label[i].label], c="r")
    
    return fig, ax


def data_distribution(df, period="M"):
    """
    绘制数据分布

    Args:
        df: pd.DataFrame
        period: str
    """
    distribution = pd.DataFrame(np.empty((len(df), 1)), 
                                index=df.index.asfreq(period))
    distribution.groupby("date").count().plot(kind='bar', legend=False)


def plot_history(model, figsize=(8, 4)):
    """
    训练历史可视化(只针对autoencoder)

    Args:
        model: GearboxFaultDetectionModel
        figsize: tuple, 图大小
    """
    history = model.detectors["auto_encoder"].history_

    set_figsize(figsize)
    
    fig, ax = plt.subplots()
    ax.plot(history['loss'])
    ax.set_title('Model loss')
    ax.set_ylabel('Loss')
    ax.set_xlabel('Epoch')
    ax.legend(['Train'], loc='upper right')
    return fig, ax


def set_figsize(figsize=(3.5, 2.5)):
    '''
    设置图的尺寸
    
    Args:
        figsize: tuple(float)
    '''
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')