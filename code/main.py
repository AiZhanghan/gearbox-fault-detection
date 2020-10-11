"""
调动所有风场所有风机数据
"""


import os

import toolkit
from Reader import Reader


def main():
    feature_path = r"D:/Workspace/python_workspace/gearbox-fault-detection/local/feature/"
    speed_path = r"D:/Workspace/python_workspace/gearbox-fault-detection/local/rotating_speed"
    # 风场
    wind_farms = os.listdir(feature_path)
    # 传感器
    sensors = [
        "gearbox",
        "low_speed_shaft",
        "high_speed_shaft",
    ]
    reader = Reader()
    for wind_farm in wind_farms:
        # 风机
        print("wind_farm: %s" % wind_farm)
        wind_turbines = os.listdir(os.path.join(feature_path, wind_farm))
        for wind_turbine in wind_turbines:
            print("wind_turbine: %s" % wind_turbine)
            # 读取数据
            feature = reader.read_feature(os.path.join(feature_path, wind_farm,
                wind_turbine), sensors)
            speed = reader.read_speed(os.path.join(speed_path, wind_farm,
                wind_turbine), sensors)
            if wind_farm == "li_niu_ping":
                feature = feature.loc[speed.index]
            toolkit.print_shape(feature=feature, speed=speed)
            # 根据转速初步过滤异常值(极值分析)
            SPEED_THRESHOLD = 250 if wind_farm != "san_tang_hu" else 3
            feature = feature[speed.speed >= SPEED_THRESHOLD]
            speed = speed[speed.speed >= SPEED_THRESHOLD]
            


if __name__ == "__main__":
    main()
