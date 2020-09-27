import os
import sys
sys.path.append("../")
import pandas as pd

import toolkit


@toolkit.timer
def get_rotating_speed(csv_path, save_path):
    """
    获取风机转速数据

    Args:
        csv_path: str, 文件路径
        save_path: str, 保存路径
    """
    # 获取风机号
    _, shotname, _ = toolkit.get_filepath_shotname_extension(csv_path)
    wind_turbine = shotname.split("_")[0]
    print("Processing wind turbine %s..." % wind_turbine)
    # 对应测点
    dic = {
        "齿轮箱低速轴_垂直径向": "low_speed_shaft",
        "齿轮箱高速轴_垂直径向": "high_speed_shaft",
        "齿轮箱内齿圈_水平径向": "gearbox",
    }

    read_csv_kw = {
        "header": None, 
        "index_col": 0, 
        "skiprows": 1, 
        "usecols": [0, 1, 4], # 时间, 测点, 转速
        "converters": {
            1: lambda x: dic[x], # 处理测点位置(CH -> EN)
        },
    }
    
    data = pd.read_csv(csv_path, **read_csv_kw)
    # 根据测点位置排序
    data = data.sort_values(by=1)

    start = 0
    # key: 测点
    # value: 测点记录数
    for key, value in data.groupby(1).size().to_dict().items():
        end = start + value
        sub_data = data.iloc[start: end]
        sub_data = sub_data.drop(1, axis=1)
        sub_data = sub_data.sort_index()
        
        sub_data.index = pd.to_datetime(sub_data.index, format="%Y%m%d%H%M%S")
        sub_data.index.name = "date"
        sub_data.columns = ["speed"]

        if not os.path.exists(os.path.join(target_path, wind_turbine)):
            os.makedirs(os.path.join(target_path, wind_turbine))
        sub_data.to_csv(os.path.join(target_path, wind_turbine, key + ".csv"))
        start = end


if __name__ == "__main__":
    base_path = r"E:\data\海装\山西福光宁武牛家岭48MW工程项目"
    target_path = r"D:\Workspace\python_workspace\gearbox-fault-detection\local\rotating_speed\niu_jia_ling"
    file_paths = os.listdir(base_path)
    for file_path in file_paths:
        get_rotating_speed(os.path.join(base_path, file_path), target_path)
    