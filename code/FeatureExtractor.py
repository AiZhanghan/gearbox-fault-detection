"""
TODO: RuntimeWarning: invalid value encountered in double_scalars?
    提取特征时报错
"""


import os
import time
import numpy as np
import pandas as pd


class FeatureExtractor:
    """特征提取器, 从data提取对应振动信号的特征"""
    
    def run(self, data, sample_frequency):
        """
        Args:
            data: pd.DataFrame, index为采样日期, values为采样数据
            sample_frequency: int, 采样频率
        
        Return:
            feature: pd.DataFrame
        """
        # 删掉Nan的记录
        data = data.dropna()
        data.index.name = "date"
        
        time_feature = data.apply(self._time_feature, axis=1)

        self.sample_frequency = sample_frequency
        frequent_feature = data.apply(self._frequency_feature, axis=1)
        
        feature = pd.concat([time_feature, frequent_feature], axis=1)
        
        return feature

    def _time_feature(self, x):
        """提取x的时域统计特征

        Args:
            x: pd.Series
        
        Return:
            pd.Series
        """
        mean = x.mean()
        sd = x.std()
        root = (np.sum(np.sqrt(np.abs(x))) / len(x)) ** 2
        rms = np.sqrt(np.sum(x ** 2) / len(x))
        peak = np.max(np.abs(x))
        
        skewness = x.skew()
        kurtosis = x.kurt()
        crest = peak / rms
        clearance = peak / root
        shape = rms / (np.sum(np.abs(x)) / len(x))
        
        impluse = peak / (np.sum(np.abs(x)) / len(x))
    
        feature = pd.Series([mean, sd, root, rms, peak, 
                             skewness, kurtosis, crest, clearance, shape, 
                             impluse], 
                            index = ['mean', 'sd', 'root', 'rms', 'peak', 
                                     'skewness', 'kurtosis', 'crest', 
                                     'clearance', 'shape', 'impluse'])
        
        return feature
    
    def _frequency_feature(self, y):
        """提取y的频域统计特征

        Args:
            y: pd.Series
        
        Return:
            pd.Series
        """
        
        # 采样点数
        N = len(y)
        # 需留意采样频率
        sample_frequency = self.sample_frequency
        # 传感器采样周期
        T = 1 / sample_frequency 
        #快速傅里叶变换,取模
        yf = 2.0 / N * np.abs(np.fft.fft(y))
        #由于对称性，只取一半区间
        yf = yf[: N // 2]  
        xf = np.linspace(0.0, 1.0 / (2.0 * T), N // 2)
        # 频域常用统计特征
        K = len(yf)
        # spectrum line
        s = yf 
        # frequency value
        f = xf 
        
        p1 = np.sum(s) / K
        p2 = np.sum((s - p1) ** 2) / (K - 1)
        p3 = np.sum((s - p1) ** 3) / (K * (np.sqrt(p2) ** 3))
        p4 = np.sum((s - p1) ** 4) / (K * p2 ** 2)
        p5 = np.sum(f * s) / np.sum(s)
        p6 = np.sqrt(np.sum((f - p5) ** 2 * s) / K)
        p7 = np.sqrt(np.sum(f ** 2 * s) / np.sum(s))
        p8 = np.sqrt(np.sum(f ** 4 * s) / np.sum(f ** 2 * s))
        p9 = np.sum(f ** 2 * s) / np.sqrt(np.sum(s) * np.sum(f ** 4 * s))
        p10 = p6 / p5
        p11 = np.sum((f - p5) ** 3 * s) / (K * p6 ** 3)
        p12 = np.sum((f - p5) ** 4 * s) / (K * p6 ** 4)
        p13 = np.sum(np.sqrt(np.abs(f - p5)) * s) / (K * np.sqrt(p6))
        p14 = np.sqrt(np.sum((f - p5) ** 2 * s) / np.sum(s))
        
        feature = pd.Series([p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, 
                             p12, p13, p14],
                            index = ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7',
                                     'p8', 'p9', 'p10', 'p11', 'p12', 'p13',
                                     'p14'])
        
        return feature
    

if __name__ == "__main__":
    # RuntimeWarning: invalid value encountered in double_scalars?
    # np.seterr(invalid="ignore")

    base_path = r"D:\Workspace\Data\牛家岭"
    wind_turbines = [
        "1",
        "2",
        "3",
    ]
    target_path = r"D:\Workspace\python_workspace\gearbox-fault-detection\local\feature\niu_jia_ling"

    sample_frequency = {
        "gearbox": 6400,
        "low_speed_shaft": 12800,
        "high_speed_shaft": 25600,
    }

    extractor = FeatureExtractor()

    for wind_turbine in wind_turbines:
        sensors = os.listdir(os.path.join(base_path, wind_turbine))
        for sensor in sensors:
            start = time.time()
            print("Extracting %s: %s." % (wind_turbine, sensor), end="\t")

            data = pd.read_csv(os.path.join(base_path, wind_turbine, sensor),
                header=None, index_col=0, parse_dates=True)

            name = slice(0, -4)
            feature = extractor.run(data, sample_frequency[sensor[name]])

            if not os.path.exists(os.path.join(target_path, wind_turbine)):
                os.makedirs(os.path.join(target_path, wind_turbine))
            feature.to_csv(os.path.join(target_path, wind_turbine, sensor))
            
            end = time.time()
            print("Cost %.2f seconds" % (end - start))