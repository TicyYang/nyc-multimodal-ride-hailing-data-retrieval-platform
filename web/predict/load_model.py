from xgboost import XGBRegressor
import pandas as pd
from pathlib import Path

directory = Path(__file__).resolve().parent

def predict_demand(year,month,day,PULocationID):
    model = XGBRegressor()
    model.load_model(directory/'xgboost_regression_model(var).model')
    data = []
    names = range(3)
    hours = range(24)
    datetime = pd.to_datetime(f'{year}-{month}-{day}')
    weekday = datetime.weekday()
    df_temp = pd.read_csv(directory/'taxi_zone_lat_lon/organized_taxi_zone_lat_lon.csv')
    lat = df_temp.loc[df_temp['LocationID'] == PULocationID, 'lat'].values[0]
    lon = df_temp.loc[df_temp['LocationID'] == PULocationID, 'lon'].values[0]

    for name in names:
        for hour in hours:
            data.append([name, year, month, day, hour, PULocationID, weekday, lat, lon])
    column_names = ['Name', 'Year', 'Month', 'Day', 'Hour', 'PULocationID', 'weekday', 'lat', 'lon']
    data = pd.DataFrame(data, columns=column_names)
    return model.predict(data)