"""
调动所有风场所有风机数据
"""


import os
import pandas as pd

import toolkit
import visualization
from Reader import Reader
from model_entry import OutlierDetector


TRAIN = {
    "li_niu_ping": {
        "1": ["2018-06", "2019-08"],
        "2": ["2018-04", "2019-08"],
        "3": ["2018-09", "2020-04"],
        "4": ["2018-06", "2019-10"],
        "5": ["2018-04", "2019-08"],
        "6": ["2018-06", "2019-08"],
        "7": ["2018-04", "2019-07"],
        "8": ["2018-04", "2019-08"],
        "9": ["2018-06", "2019-08"],
        "10": ["2018-06", "2019-08"],
        "11": ["2018-04", "2019-08"],
        "12": ["2018-06", "2019-08"],
        "13": ["2018-06", "2019-10"],
        "14": ["2018-04", "2019-10"],
        "15": ["2018-04", "2019-08"],
        "16": ["2018-04", "2019-08"],
        "17": ["2018-04", "2019-09"],
        "18": ["2018-04", "2019-08"],
        "19": ["2018-04", "2019-10"],
        "20": ["2018-04", "2019-08"],
        "21": ["2018-04", "2019-09"],
        "22": ["2018-04", "2019-10"],
        "23": ["2018-04", "2019-08"],
        "24": ["2018-04", "2019-09"],
        "25": ["2018-04", "2019-10"],
    },
    "niu_jia_ling": {
        "1": ["2018-05", "2019-06"],
        "2": ["2018-05", "2019-05"],
        "3": ["2018-05", "2019-06"],
        "4": ["2018-05", "2019-06"],
        "5": ["2018-05", "2019-06"],
        "6": ["2018-05", "2019-06"],
        "7": ["2018-05", "2019-07"],
        "8": ["2018-05", "2019-06"],
        "9": ["2018-05", "2019-06"],
        "10": ["2018-05", "2019-06"],
        "11": ["2019-04", "2019-11"],
        "12": ["2018-05", "2019-05"],
        "13": ["2018-05", "2019-06"],
        "14": ["2018-05", "2019-02"],
        "15": ["2018-05", "2019-06"],
        "16": ["2018-05", "2019-06"],
        "17": ["2018-05", "2019-08"],
        "18": ["2018-05", "2019-06"],
        "20": ["2018-05", "2019-06"],
        "21": ["2018-05", "2019-06"],
        "22": ["2018-05", "2019-06"],
        "23": ["2018-05", "2019-06"],
        "24": ["2018-05", "2019-06"],
    },
    "san_tang_hu": {
        "5": ["2018-04", "2018-07"],
        "29": ["2018-01", "2018-05"],
        "33": ["2018-01", "2018-03"],
        "40": ["2018-04", "2018-07"],
        "50": ["2018-01", "2018-04"],
        "58": ["2018-04", "2018-07"],
        "66": ["2018-01", "2018-03"],
        "73": ["2018-01", "2018-04"],
        "76": ["2018-01", "2018-04"],
        "86": ["2018-01", "2018-04"],
        "89": ["2018-01", "2018-04"],
        "96": ["2018-04", "2018-06"],
    },
}
TEST = {
    "li_niu_ping": {
        "1": ["2019-09", "2020-04"],
        "2": ["2019-09", "2020-04"],
        "3": ["2018-06", "2018-08"],
        "4": ["2019-11", "2020-04"],
        "5": ["2019-09", "2020-04"],
        "6": ["2019-09", "2020-04"],
        "7": ["2019-08", "2020-04"],
        "8": ["2019-09", "2020-04"],
        "9": ["2019-09", "2020-04"],
        "10": ["2019-09", "2020-04"],
        "11": ["2019-09", "2020-04"],
        "12": ["2019-09", "2020-04"],
        "13": ["2019-11", "2020-04"],
        "14": ["2019-11", "2020-04"],
        "15": ["2019-09", "2020-04"],
        "16": ["2019-09", "2020-04"],
        "17": ["2019-10", "2020-04"],
        "18": ["2019-09", "2020-04"],
        "19": ["2019-11", "2020-04"],
        "20": ["2019-09", "2020-04"],
        "21": ["2019-10", "2020-04"],
        "22": ["2019-11", "2020-04"],
        "23": ["2019-09", "2020-04"],
        "24": ["2019-10", "2020-04"],
        "25": ["2019-11", "2020-04"],
    },
    "niu_jia_ling": {
        "1": ["2019-07", "2019-11"],
        "2": ["2019-06", "2019-10"],
        "3": ["2019-07", "2019-11"],
        "4": ["2019-07", "2019-11"],
        "5": ["2019-07", "2019-11"],
        "6": ["2019-07", "2019-11"],
        "7": ["2019-08", "2019-11"],
        "8": ["2019-07", "2019-11"],
        "9": ["2019-07", "2019-11"],
        "10": ["2019-07", "2019-11"],
        "11": ["2018-05", "2018-12"],
        "12": ["2019-06", "2019-07"],
        "13": ["2019-07", "2019-11"],
        "14": ["2019-03", "2019-08"],
        "15": ["2019-07", "2019-11"],
        "16": ["2019-07", "2019-11"],
        "17": ["2019-09", "2019-11"],
        "18": ["2019-07", "2019-11"],
        "20": ["2019-07", "2019-11"],
        "21": ["2019-07", "2019-11"],
        "22": ["2019-07", "2019-11"],
        "23": ["2019-07", "2019-11"],
        "24": ["2019-07", "2019-11"],
    },
    "san_tang_hu": {
        "5": ["2018-08", "2018-08"],
        "29": ["2018-06", "2018-06"],
        "33": ["2018-04", "2018-04"],
        "40": ["2018-08", "2018-08"],
        "50": ["2018-05", "2018-08"],
        "58": ["2018-08", "2018-08"],
        "66": ["2018-04", "2018-05"],
        "73": ["2018-05", "2018-08"],
        "76": ["2018-05", "2018-08"],
        "86": ["2018-05", "2018-05"],
        "89": ["2018-05", "2018-05"],
        "96": ["2018-08", "2018-08"],
    },
}


def main():
    feature_path = r"D:/Workspace/python_workspace/gearbox-fault-detection/local/feature/"
    speed_path = r"D:/Workspace/python_workspace/gearbox-fault-detection/local/rotating_speed"
    result_path = r"D:/Workspace/python_workspace/gearbox-fault-detection/code/result"
    # 风场
    # wind_farms = os.listdir(feature_path)
    wind_farms = [
        # "li_niu_ping",
        "niu_jia_ling",
        # "san_tang_hu",
    ]
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
            # if wind_turbine != "11":
            #     continue
            print("wind_turbine: %s" % wind_turbine)
            # 读取数据
            feature = reader.read_feature(os.path.join(feature_path, 
                wind_farm, wind_turbine), sensors)
            speed = reader.read_speed(os.path.join(speed_path, wind_farm,
                wind_turbine), sensors)
            # 只有犁牛坪需要对齐feature speed
            if wind_farm == "li_niu_ping":
                feature = feature.loc[speed.index]
            toolkit.print_shape(feature=feature, speed=speed)
            # 根据转速初步过滤异常值(极值分析), 三塘湖转速单位不同
            SPEED_THRESHOLD = 250 if wind_farm != "san_tang_hu" else 3
            feature = feature[speed.speed >= SPEED_THRESHOLD]
            speed = speed[speed.speed >= SPEED_THRESHOLD]
            # 划分训练集和测试集
            train_start, train_end = TRAIN[wind_farm][wind_turbine]
            feature_train = feature[train_start: train_end]

            test_start, test_end = TEST[wind_farm][wind_turbine]
            feature_test = feature[test_start: test_end]

            toolkit.print_shape(feature_train=feature_train,
                feature_test=feature_test)
            
            feature_test = pd.concat([feature_train, feature_test]).sort_index()
            # 训练
            detector = OutlierDetector()
            detector.fit(feature_train, contamination=0.01)
            anomaly_scores_train = detector.decision_scores
            label_train = detector.label
            # 测试
            anomaly_scores_test = detector.decision_function(feature_test)
            label_test = detector.predict(feature_test)

            # 可视化结果
            fig, _ = visualization.plot_line(anomaly_scores_train, label_train,
                anomaly_scores_test, label_test, detector.threshold, wind_farm, 
                wind_turbine)
            
            temp = os.path.join(result_path, wind_farm)
            if not os.path.exists(temp):
                os.makedirs(temp)
            fig.savefig(os.path.join(temp, wind_turbine + ".png"))
            # 保存报警记录
            record = anomaly_scores_test[label_test.label]
            if len(record) > 0:
                record.to_csv(os.path.join(temp, wind_turbine + ".csv"))


if __name__ == "__main__":
    main()
