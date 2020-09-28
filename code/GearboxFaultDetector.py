import pandas as pd

import toolkit
import model_entry


class GearboxFaultDetector:
    """
    根据转速工况, 分成3个子模型
    """
    def __init__(self, low_speed=1000, mid_speed=1250, high_speed=1500):
        """
        Args:
            low_speed: 1000
            mid_speed: 1250
            high_speed: 1500
        """
        self.low_speed = low_speed
        self.mid_speed = mid_speed
        self.high_speed = high_speed

        self.low_outlier_detector = model_entry.OutlierDetector()
        self.mid_outlier_detector = model_entry.OutlierDetector()
        self.high_outlier_detector = model_entry.OutlierDetector()

        self.decision_scores = None

    def fit(self, feature, speed):
        """
        Args:
            feature: pd.DataFrame
            speed: pd.DataFrame
        """
        low_feature, mid_feature, high_feature = self._split_data(feature, 
            speed)
        # fit
        self.low_outlier_detector.fit(low_feature)
        self.mid_outlier_detector.fit(mid_feature)
        self.high_outlier_detector.fit(high_feature)
        # 整合decision_scores
        self.decision_scores = pd.concat([
            self.low_outlier_detector.decision_scores,
            self.mid_outlier_detector.decision_scores, 
            self.high_outlier_detector.decision_scores]).sort_index()
        self.decision_scores.columns = ["score"]
    
    def decision_function(self, feature, speed):
        """
        Args:
            feature: pd.DataFrame
            speed: pd.DataFrame
        
        Return:
            pd.DataFrame
        """
        low_feature, mid_feature, high_feature = self._split_data(feature, 
            speed)
        # predict
        anomaly_scores = pd.concat([
            self.low_outlier_detector.decision_function(low_feature),
            self.mid_outlier_detector.decision_function(mid_feature),
            self.high_outlier_detector.decision_function(high_feature)])\
            .sort_index()
        return anomaly_scores

    def _split_data(self, feature, speed):
        """
        Args:
            feature: pd.DataFrame
            speed: pd.DataFrame
        
        Return:
            tuple(pd.DataFrame)
        """
        # 剔除小于low_speed的数据
        feature = feature[speed.speed >= self.low_speed]
        speed = speed[speed.speed >= self.low_speed]
        # 子工况数据
        low_feature = feature[feature.index.isin(\
            speed.query("speed < %d" % self.mid_speed).index)]
        mid_feature = feature[feature.index.isin(\
            speed.query("speed >= %d & speed < %d" %\
            (self.mid_speed, self.high_speed)).index)]
        high_feature = feature[feature.index.isin(\
            speed.query("speed >= %d" % self.high_speed).index)]
        toolkit.print_shape(feature=feature, low_feature=low_feature,
            mid_feature=mid_feature, high_feature=high_feature)
        return low_feature, mid_feature, high_feature


if __name__ == "__main__":
    model = GearboxFaultDetector()