import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
# from sklearn.decomposition import PCA
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.combination import average
from pyod.models.combination import aom
from pyod.utils.utility import standardizer

import toolkit


class OutlierDetector:
    """
    Attr:
        decision_scores: pd.DataFrame, 训练集异常值
        label: pd.DataFrame, 训练集预测标签
        threshold: float, 阈值
    
    Instance API:
        fit(X, contamination): 拟合模型
        decision_function(X): 使用拟合好的模型预测
        predict(X)
        # load(path)

    Static API:
        # save(path)
    """
    def __init__(self):
        self.decision_scores = None
        self.label = None
        self.threshold = None
        # 模型
        self.detectors = None
        # data scaler
        self.data_norm_scalar = None
        self.data_unif_scalar = None
        # reducer
        self.reducer = None
        # score scaler
        self.score_scalar = None

    def fit(self, X, contamination=0.1, detector_num=4):
        """
        Fit detector

        Args:
            X: pd.DataFrame
        """
        self.detectors = {}
        for i in range(detector_num):
            self.detectors[i] = AutoEncoder(
                # epochs=256,
                validation_size=0,
                preprocessing=False,
                verbose=0,
                # contamination=contamination,
            )
        # 数据预处理
        X_train = self.data_preprocess_fit_transform(X)
        # 降维
        # X_train = self.reduction_fit_transform(X_train)
        train_scores = np.zeros([X.shape[0], len(self.detectors)])
        # 训练
        for i, clf_name in enumerate(self.detectors):
            clf = self.detectors[clf_name]
            clf.fit(X_train)
            train_scores[:, i] = clf.decision_scores_
        # 训练集异常程度及阈值
        train_scores_norm, self.score_scalar = standardizer(train_scores, 
            keep_scalar=True)
        
        if detector_num == 1:
            self.decision_scores = pd.DataFrame(average(train_scores_norm),
                index=X.index)
        else:
            self.decision_scores = pd.DataFrame(aom(train_scores_norm, 
                n_buckets=round(detector_num ** 0.5)), index=X.index)
        self.decision_scores.columns = ["score"]
        self.threshold = self.decision_scores.quantile(1 - contamination)[0]
        self.label = self.get_label(self.decision_scores)
        
    def decision_function(self, X):
        """
        Predict raw anomaly score of X using the fitted detector.

        Args:
            X: pd.DataFrame
        
        Return:
            anomaly_scores: pd.DataFrame
        """
        # 数据预处理
        X_test = self.data_preprocess_transform(X)
        # 降维
        # X_test = self.reduction_transform(X_test)
        test_scores = np.zeros([X_test.shape[0], len(self.detectors)])
        for i, clf_name in enumerate(self.detectors):
            test_scores[:, i] = self.detectors[clf_name].\
                decision_function(X_test)
        
        test_scores_norm = self.score_scalar.transform(test_scores)
        anomaly_scores = pd.DataFrame(average(test_scores_norm), 
            index=X.index)
        anomaly_scores.columns = ["score"]

        return anomaly_scores

    def predict(self, X):
        """

        Args:
            X: pd.DataFrame
        
        Return:
            pd.DataFrame
        """
        anomaly_scores = self.decision_function(X)
        # 连续三次超过阈值
        return self.get_label(anomaly_scores)
    
    def data_preprocess_fit_transform(self, X):
        """
        Args:
            X: pd.DataFrame
        
        Return:
            np.array
        """
        # 数据预处理
        # 标准化
        X_train_norm, self.data_norm_scalar = standardizer(X, 
            keep_scalar=True)
        # 归一化
        X_train_unif, self.data_unif_scalar = minmaxizer(X_train_norm, 
            keep_scalar=True)
        return X_train_unif
    
    def data_preprocess_transform(self, X):
        """
        Args:
            X: pd.DataFrame
        
        Return:
            np.array
        """
        X_test_norm = self.data_norm_scalar.transform(X)   
        X_test_unif = self.data_unif_scalar.transform(X_test_norm)
        return X_test_unif

    # def reduction_fit_transform(self, X):
    #     """
    #     Args:
    #         X: np.array
        
    #     Return:
    #         np.array
    #     """
    #     self.reducer = PCA(n_components=8)
    #     return self.reducer.fit_transform(X)

    # def reduction_transform(self, X):
    #     """
    #     Args:
    #         X: np.array
        
    #     Return:
    #         np.array
    #     """
    #     return self.reducer.transform(X)
    
    def get_label(self, anomaly_scores):
        """
        Args:
            anomaly_scores: pd.DataFrame
        
        Return:
            pd.DataFrame
        """
        label = (anomaly_scores >= self.threshold).astype(int).rolling(3).\
            sum() == 3
        label.rename(columns={"score": "label"}, inplace=True)
        return label
    
    # def save(self, path):
    #     """
    #     Args:
    #         path: str
    #     """
    #     base_path, _, _ = toolkit.get_filepath_shotname_extension(path)
    #     if not os.path.exists(base_path):
    #         os.makedirs(base_path)
    #     with open(path, "wb") as f:
    #         pickle.dump(self, f)

    # @staticmethod
    # def load(path):
    #     """
    #     Args:
    #         path: str
        
    #     Return:
    #         Model
    #     """
    #     with open(path, "rb") as f:
    #         model = pickle.load(f)
    #     return model


def minmaxizer(X, X_t=None, keep_scalar=False):
    """归一化

    Args:
        X: np.array, (n_samples, n_features), 训练集
        X_t: np.array, (n_samples_new, n_features), 待转换数据
        keep_scalar: bool

    Return:
        X_unif: np.array
        X_t_unif: np.array
        scalar : sklearn scalar object
    """
    scaler = MinMaxScaler().fit(X)

    if X_t is None:
        if keep_scalar:
            return scaler.transform(X), scaler
        else:
            return scaler.transform(X)
    else:
        if X.shape[1] != X_t.shape[1]:
            raise ValueError(
                "The number of input data feature should be consistent"
                "X has {0} features and X_t has {1} features.".format(
                    X.shape[1], X_t.shape[1]))
        if keep_scalar:
            return scaler.transform(X), scaler.transform(X_t), scaler
        else:
            return scaler.transform(X), scaler.transform(X_t)
            