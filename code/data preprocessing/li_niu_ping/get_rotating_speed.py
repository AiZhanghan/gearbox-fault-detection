import os
import pandas as pd


def get_rotating_speed(scada_path, feature_path, target_path, wind_turbine,
    sensor):
    """
    Args:
        scada_path: str
        feature_path: str
        target_path: str
        wind_turbine: str
        sensor: str
    """
    # "001" -> "1"
    wind_turbine_fix = str(int(wind_turbine))
    print("%s %s" % (wind_turbine_fix, sensor))
    
    feature = pd.read_csv(os.path.join(feature_path, wind_turbine_fix, 
        sensor + ".csv"), index_col=0, parse_dates=True)
    feature.index = feature.index.to_period("min")

    scada_file = os.listdir(os.path.join(scada_path, wind_turbine))[0]
    scada = pd.read_csv(os.path.join(scada_path, wind_turbine, scada_file), 
        encoding="gbk", index_col=0, parse_dates=True)
    scada.index = scada.index.to_period("min")
    
    speed = scada.loc[scada.index.intersection(feature.index)]["发电机转速"]
    speed = speed.to_frame()
    speed.index.name = "date"
    speed.columns = ["speed"]

    if not os.path.exists(os.path.join(target_path, wind_turbine_fix)):
        os.makedirs(os.path.join(target_path, wind_turbine_fix))
    speed.to_csv(os.path.join(target_path, wind_turbine_fix, sensor + ".csv"))


def main():
    scada_path = r"D:\Workspace\CSIC\tool\data_exploring_platform\SCADA_HISTORY\中能建投广东韶关南雄犁牛坪50MW工程项目"
    feature_path = r"D:\Workspace\python_workspace\gearbox-fault-detection\local\feature\li_niu_ping"
    target_path = r"D:\Workspace\python_workspace\gearbox-fault-detection\local\rotating_speed\li_niu_ping"
    wind_turbines = os.listdir(scada_path)
    sensors = [
        "gearbox",
        "low_speed_shaft",
        "high_speed_shaft",
    ]

    for wind_turbine in wind_turbines:
        for sensor in sensors:
            get_rotating_speed(scada_path, feature_path, target_path, 
                wind_turbine, sensor)


if __name__ == "__main__":
    main()