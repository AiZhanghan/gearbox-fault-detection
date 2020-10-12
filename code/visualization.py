import numpy as np
import pandas as pd
from IPython import display
from matplotlib import pyplot as plt


def plot_line(anomaly_scores_train, anomaly_scores_test, wind_farm, 
    wind_turbine):
    """
    Args:
        anomaly_scores_train: pd.DataFrame
        anomaly_scores_test: pd.DataFrame
        wind_farm: str
        wind_turbine: str
    """
    set_figsize(figsize=(8, 8))
    fig, ax = plt.subplots(nrows=2, ncols=1, constrained_layout=True)

    anomaly_scores_train.plot(ax=ax[0])
    ax[0].set_title("%s #%s train" % (wind_farm, wind_turbine))
    
    anomaly_scores_test.plot(ax=ax[1])
    ax[1].set_title("%s #%s test" % (wind_farm, wind_turbine))
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