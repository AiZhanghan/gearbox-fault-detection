import os
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from pyod.models.auto_encoder import AutoEncoder
from pyod.models.combination import average
from pyod.utils.utility import standardizer

import toolkit


class OutlierDetector:
    """
    Attr:
        decision_scores: pd.DataFrame, 训练集异常值
    
    Instance API:
        fit(X): 拟合模型
        decision_function(X): 使用拟合好的模型预测
        load(path)

    Static API:
        save(path)
    """

    def fit(self, X):
        """Fit detector

        Args:
            X: pd.DataFrame
        """
        self.detectors = {
            "auto_encoder": AutoEncoder(
                epochs=256,
                validation_size=0,
                preprocessing=False,
                verbose=0,
            ),
        }
        # print("train_data.shape:", X.shape)
        # 数据预处理
        # 标准化
        X_train_norm, self.data_norm_scalar = standardizer(X, 
            keep_scalar=True)
        # 归一化
        X_train_unif, self.data_unif_scalar = minmaxizer(X_train_norm, 
            keep_scalar=True)
        
        train_scores = np.zeros([X.shape[0], len(self.detectors)])

        for i, clf_name in enumerate(self.detectors):
            clf = self.detectors[clf_name]
            clf.fit(X_train_unif)
            train_scores[:, i] = clf.decision_scores_
        
        train_scores_norm, self.score_scalar = standardizer(train_scores, 
            keep_scalar=True)

        self.decision_scores = pd.DataFrame(average(train_scores_norm),
            index=X.index)
    
    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        Args:
            X: pd.DataFrame
        
        Return:
            anomaly_scores: pd.DataFrame
        """
        # 数据预处理
        X_test_norm = self.data_norm_scalar.transform(X)
        X_test_unif = self.data_unif_scalar.transform(X_test_norm)

        test_scores = np.zeros([X_test_unif.shape[0], len(self.detectors)])
        for i, clf_name in enumerate(self.detectors):
            test_scores[:, i] = \
                self.detectors[clf_name].decision_function(X_test_unif)
        
        test_scores_norm = self.score_scalar.transform(test_scores)
        anomaly_scores = pd.DataFrame(average(test_scores_norm), 
            index=X.index)

        return anomaly_scores
    
    def save(self, path):
        """
        Args:
            path: str
        """
        base_path, _, _ = toolkit.get_filepath_shotname_extension(path)
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path):
        """
        Args:
            path: str
        
        Return:
            Model
        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model


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
            