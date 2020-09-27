import os
import pandas as pd


class Reader:
    def read_feature(self, wind_turbine_path, sensors):
        """
        从wind_turbine_path读取该风机sensors对应特征并进行规整

        Args:
            wind_turbine_path: str
            sensors: list[str], 传感器
        
        Retuen:
            pd.DataFrame
        """
        return self.read(wind_turbine_path, sensors)
    
    def read_speed(self, wind_turbine_path, sensors):
        """
        从wind_turbine_path读取该风机sensors对应风速并进行规整

        Args:
            wind_turbine_path: str
            sensors: list[str], 传感器
        
        Retuen:
            pd.DataFrame
        """
        speeds = self.read(wind_turbine_path, sensors)
        speed = speeds.mean(axis=1).to_frame()
        speed.columns = ["speed"]
        return speed

    def read(self, wind_turbine_path, sensors):
        """
        Args:
            wind_turbine_path: str
            sensors: list[str], 传感器
        
        Retuen:
            pd.DataFrame
        """
        datas = {}
        for sensor in sensors:
            path = os.path.join(wind_turbine_path, sensor + '.csv')
            datas[sensor] = pd.read_csv(path, index_col=0, parse_dates=True)
        # 重采样
        self._resample(datas)
        # 规整到一个DataFrame, join="inner"保证对齐, index取交集
        datas = pd.concat(datas, axis=1, join="inner")
        datas = datas.dropna()
        return datas

    def _resample(self, datas):
        """重采样, 由原来的秒 -> 时

        Args:
            datas: dict{sensor: data}
        """
        for sensor in datas:
            # 不同传感器采样时刻有些许差别, 在同一小时内, 算一次采样
            datas[sensor].index = datas[sensor].index.to_period('H')
            # 若某传感器在一小时内采样多次, 对多次采样特征取均值
            datas[sensor] = datas[sensor].groupby("date").mean()
