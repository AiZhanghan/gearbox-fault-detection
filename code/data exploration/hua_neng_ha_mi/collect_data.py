"""
处理原始csv文件
"""


import os
import time
import collections
import pandas as pd


def group_by_sensor(files):
    """
    根据传感器测点位置以及采样频率进行分组,

    Args:
        files: list[str]
    
    Return:
        dict
    """
    # print("len(files): %d", len(files))
    res = collections.defaultdict(list)
    for file in files:
        sensor = get_sensor_info(file)
        res[sensor].append(file)
    # 输出每个测点的数据数量
    # for sensor in res:
    #     print("%s: %d" % (sensor, len(res[sensor])))
    return res


def get_sensor_info(file):
    """
    Args:
        file: str
    
    Return:
        str
    """
    res = None
    file_list = file.split("_")
    if file_list[2] == "GEN":
        res = "_".join(file_list[2: 6])
    else:
        res = "_".join(file_list[2: 5])
    return res


def filter_by_sensor(files, sensors):
    """
    过滤

    Args:
        files: dict
        snesor: list[str]
    
    Return:
        dict
    """
    res = {}
    for key in files:
        if key in sensors:
            res[key] = files[key]
    return res


def collect_data(files, wind_turbine, source_path, target_path):
    """
    聚合成一个csv文件, 只要2018年之后的数据

    Args:
        files: list[str]
        wind_turbine: str
        source_path: str
        target_path: str
    """
    dic = {
        "Planet_径向_12800": "gearbox", 
        "Shaft2_径向_25600": "low_speed_shaft", 
        "Shaft3_径向_25600": "high_speed_shaft",
    }
    for sensor in files:
        start = time.time()
        print("%s: " % sensor, end="")
        df = to_DataFrame(files[sensor])
        # 读取csv文件
        data = pd.DataFrame()
        for path in df.values:
            # np.array -> str
            path = path[0]
            temp = pd.read_csv(os.path.join(source_path, wind_turbine, path), 
                header=None)
            data = pd.concat([data, temp])
        
        data.index = df.index
        data.dropna(axis=1, inplace=True)
        # 保存csv文件
        temp_path = os.path.join(target_path, wind_turbine)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        data.to_csv(os.path.join(temp_path, dic[sensor]) + ".csv", header=0)
        print("%.2f" % (time.time() - start))


def collect_speed(files, wind_turbine, source_path, target_path):
    """
    Args:
        files: list[str]
        wind_turbine: str
        source_path: str
        target_path: str
    """
    dic = {
        "Planet_径向_12800": "gearbox", 
        "Shaft2_径向_25600": "low_speed_shaft", 
        "Shaft3_径向_25600": "high_speed_shaft",
    }
    for sensor in files:
        print("%s: " % sensor)
        df = to_DataFrame(files[sensor])
        speed = df.apply(lambda x: x[0].split("_")[6][: -3], axis=1)
        speed.index = df.index
        # 保存csv文件
        temp_path = os.path.join(target_path, wind_turbine)
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        speed.to_csv(os.path.join(temp_path, dic[sensor]) + ".csv", header=0)


def to_DataFrame(files):
    """
    Args:
        files: list[str]
    
    Return:
        pd.DataFrame
    """
    df = pd.DataFrame(files)
    df.index = pd.to_datetime(df.apply(lambda x: x[0].split("_")[-1][: -4],
        axis=1))
    df.sort_index(inplace=True)
    df = df["2018": ]
    return df


def main():
    source_path = r"E:\data\海装\CMS\华能三塘湖"
    # target_path = r"E:\data\海装\华能新疆哈密风电基地200MW风电工程项目"
    target_path = r"D:\Workspace\python_workspace\gearbox-fault-detection\local\rotating_speed\san_tang_hu"
    # wind_turbines = os.listdir(source_path)
    wind_turbines = [
        "5",
        "29",
        "33",
        "40",
        "50",
        "58",
        "66",
        "73",
        "76",
        "86",
        "89",
        "96",
    ]
    
    for wind_turbine in wind_turbines:
        print("\nwind_turbine: %s" % wind_turbine)
        files = os.listdir(os.path.join(source_path, wind_turbine))
        files = group_by_sensor(files)
        sensors = ["Planet_径向_12800", "Shaft2_径向_25600", "Shaft3_径向_25600"]
        files = filter_by_sensor(files, sensors)
        # collect_data(files, wind_turbine, source_path, target_path)
        collect_speed(files, wind_turbine, source_path, target_path)


if __name__ == "__main__":
    main()
