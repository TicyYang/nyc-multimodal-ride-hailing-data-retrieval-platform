from holidays.countries import US
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')

fhvhv_202011 = pd.read_parquet('../fhvhv_tripdata_2020-11.parquet', engine='pyarrow')
fhvhv_202012 = pd.read_parquet('../fhvhv_tripdata_2020-12.parquet', engine='pyarrow')
fhvhv = pd.concat([fhvhv_202011, fhvhv_202012]).reset_index(drop=True)
uber = fhvhv[fhvhv.hvfhs_license_num == 'HV0003'].sort_values('pickup_datetime').reset_index(drop=True)
lyft = fhvhv[fhvhv.hvfhs_license_num == 'HV0005'].sort_values('pickup_datetime').reset_index(drop=True)


yellow_202011 = pd.read_parquet('../yellow_tripdata_2020-11.parquet', engine='pyarrow')
yellow_202012 = pd.read_parquet('../yellow_tripdata_2020-12.parquet', engine='pyarrow')
yellow = pd.concat([yellow_202011, yellow_202012]).sort_values('tpep_pickup_datetime').reset_index(drop=True)


# yellow_wrong_time = yellow_202012[yellow_202012['tpep_pickup_datetime'].dt.to_period('M') != '2020-12']
# yellow_wrong_time = yellow_wrong_time.sort_values(by='tpep_pickup_datetime').reset_index(drop=True)
# print(f'Num of yellow wrong pickup datetime: {yellow_wrong_time.shape[0]}')

# yellow_wrong_time_1130 = yellow_wrong_time[yellow_wrong_time['tpep_pickup_datetime'].dt.to_period('D') == '2020-11-30']
# print(f'Num of yellow wrong pickup datetime at 2020-11-30: {yellow_wrong_time_1130.shape[0]}')
# yellow_wrong_time_0101 = yellow_wrong_time[yellow_wrong_time['tpep_pickup_datetime'].dt.to_period('D') == '2021-01-01']
# print(f'Num of yellow wrong pickup datetime at 2021-01-01: {yellow_wrong_time_0101.shape[0]}')






green_202011 = pd.read_parquet('../green_tripdata_2020-11.parquet', engine='pyarrow')
green_202012 = pd.read_parquet('../green_tripdata_2020-12.parquet', engine='pyarrow')
green = pd.concat([green_202011, green_202012]).sort_values('lpep_pickup_datetime').reset_index(drop=True)


del fhvhv_202011, fhvhv_202012, fhvhv, yellow_202011, yellow_202012, green_202011, green_202012
# %% 查看各Dataset紀錄數
total_records = uber.shape[0] + lyft.shape[0] + yellow.shape[0] + green.shape[0]

print(f'Uber: {uber.shape[0]} {round(uber.shape[0]/total_records*100, 2)}%')
print(f'Lyft: {lyft.shape[0]} {round(lyft.shape[0]/total_records*100, 2)}%')
print(f'Yellow: {yellow.shape[0]} {round(yellow.shape[0]/total_records*100, 2)}%')
print(f'Green: {green.shape[0]} {round(green.shape[0]/total_records*100, 2)}%')
print(f'Total: {total_records}')
# Uber: 8375281
# Lyft: 3090686
# Yellow: 1509000
# Green: 88605
del total_records
# %% 取需要的column
uber = uber.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID',
                    'DOLocationID', 'trip_miles', 'trip_time',
                    'base_passenger_fare']]
lyft = lyft.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID',
                    'DOLocationID', 'trip_miles', 'trip_time',
                    'base_passenger_fare']]
yellow = yellow.loc[:, ['tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance',
                        'PULocationID', 'DOLocationID', 'fare_amount']]
green = green.loc[:, ['lpep_pickup_datetime', 'lpep_dropoff_datetime', 'trip_distance',
                      'PULocationID', 'DOLocationID', 'fare_amount']]

# %% 欄位改名
# Uber、Lyft行程時間、車資欄位改名
uber = uber.rename(columns={'trip_time': 'trip_time(s)',
                                          'base_passenger_fare': 'base_fare'})
lyft = lyft.rename(columns={'trip_time': 'trip_time(s)',
                                          'base_passenger_fare': 'base_fare'})

# 小黃、小綠上車時間、下車時間、行駛距離、車資欄位改名
yellow = yellow.rename(columns={'tpep_pickup_datetime': 'pickup_datetime',
                                              'tpep_dropoff_datetime': 'dropoff_datetime',
                                              'trip_distance': 'trip_miles',
                                              'fare_amount': 'base_fare'})
green = green.rename(columns={'lpep_pickup_datetime': 'pickup_datetime',
                                            'lpep_dropoff_datetime': 'dropoff_datetime',
                                            'trip_distance': 'trip_miles',
                                            'fare_amount': 'base_fare'})
# %% 檢查缺失值
print('-----Uber-----')
print(uber.info(show_counts=True))
print('\n-----Lyft-----')
print(lyft.info(show_counts=True))
print('\n-----Yellow-----')
print(yellow.info(show_counts=True))
print('\n-----Green-----')
print(green.info(show_counts=True))

# %% 檢查小黃、小綠的日期時間
# 小黃
# yellow_wrong_time = yellow[yellow['pickup_datetime'].dt.to_period('M') != '2020-11']
# yellow_wrong_time = yellow_wrong_time.sort_values(by='pickup_datetime').reset_index(drop=True)
# print(f'Num of yellow wrong pickup datetime: {yellow_wrong_time.shape[0]}')

# yellow_wrong_time_1031 = yellow_wrong_time[yellow_wrong_time['pickup_datetime'].dt.to_period('D') == '2020-10-31']
# print(f'Num of yellow wrong pickup datetime at 2020-10-31: {yellow_wrong_time_1031.shape[0]}')
# yellow_wrong_time_1201 = yellow_wrong_time[yellow_wrong_time['pickup_datetime'].dt.to_period('D') == '2020-12-01']
# print(f'Num of yellow wrong pickup datetime at 2020-12-01: {yellow_wrong_time_1201.shape[0]}')
# 小黃的112筆中有48筆為10月31日的資料，應可歸於10月的資料，於此月分析先剔除，與所有資料結合時可保留
# 小黃的112筆中有16筆為12月1日的資料，應可歸於12月的資料，於此月分析先剔除，與所有資料結合時可保留
# 其餘48筆日期差距較大，應予刪除
yellow = yellow[yellow['pickup_datetime'].dt.to_period('M').between('2020-11', '2020-12')]
# del yellow_wrong_time, yellow_wrong_time_1031, yellow_wrong_time_1201


# 小綠
# green_wrong_time = green[green['pickup_datetime'].dt.to_period('M') != '2020-11']
# green_wrong_time = green_wrong_time.sort_values(by='pickup_datetime').reset_index(drop=True)
# print(f'Num of green wrong pickup datetime: {green_wrong_time.shape[0]}')
# 小綠的2筆分別為前一個月最後一天接近24點，和後一個月最後一天剛過24點，應可歸於10月和12月的資料，於此月分析先剔除，與所有資料結合時可保留
green = green[green['pickup_datetime'].dt.to_period('M').between('2020-11', '2020-12')]
# del green_wrong_time
# %% 計算小黃、小綠的行程時間
yellow['trip_time(s)'] = yellow['dropoff_datetime'] - yellow['pickup_datetime']
yellow['trip_time(s)'] = round(yellow['trip_time(s)'] / np.timedelta64(1, 's'), 0)

green['trip_time(s)'] = green['dropoff_datetime'] - green['pickup_datetime']
green['trip_time(s)'] = round(green['trip_time(s)'] / np.timedelta64(1, 's'), 0)
# %% 未知上下車地點
#%%% 檢查未知上車地點佔整體資料比例
def check_unknown_pu_zone(dataset):
    unknown_pick_zone = dataset[(dataset['PULocationID'] == 264) | (dataset['PULocationID'] == 265)]
    print(unknown_pick_zone.shape[0])
    print(f'{round(unknown_pick_zone.shape[0] / dataset.shape[0] * 100, 5)}%')
    print('--------')
    return unknown_pick_zone


print('Uber num of unknown pickup zone')
check_unknown_pu_zone(uber)

print('Lyft num of unknown pickup zone')
check_unknown_pu_zone(lyft)

print('Yellow num of unknown pickup zone')
check_unknown_pu_zone(yellow)

print('Green num of unknown pickup zone')
check_unknown_pu_zone(green)

#%%% 刪除未知上車地點的records
# 定義刪除用function
def remove_unknown_pu_zone(dataset):
    '''用function "check_unknown_pu_zone"的回傳值，也就是包含未知上車地點的紀錄，作為參數'''
    dataset = dataset.drop(check_unknown_pu_zone(dataset).index, axis=0)
    dataset = dataset.reset_index(drop=True)
    return dataset


uber = remove_unknown_pu_zone(uber)
lyft = remove_unknown_pu_zone(lyft)
yellow = remove_unknown_pu_zone(yellow)
green = remove_unknown_pu_zone(green)

#%%% 檢查未知下車地點佔整體資料比例
def check_unknown_do_zone(dataset):
    unknown_drop_zone = dataset[(dataset['DOLocationID'] == 264) | (
        dataset['DOLocationID'] == 265)]
    print(unknown_drop_zone.shape[0])
    print(f'{round(unknown_drop_zone.shape[0] / dataset.shape[0] * 100, 5)}%')
    print('--------')
    return unknown_drop_zone


print('Uber num of unknown drop off zone')
check_unknown_do_zone(uber)

print('Lyft num of unknown drop off zone')
check_unknown_do_zone(lyft)

print('Yellow num of unknown drop off zone')
check_unknown_do_zone(yellow)

print('Green num of unknown drop off zone')
check_unknown_do_zone(green)

#%%% 刪除未知下車地點的records
# 未知下車地點records較多，但保留沒有意義，刪除
# 定義刪除用function
def remove_unknown_do_zone(dataset):
    '''用function "check_unknown_do_zone"的回傳值，也就是包含未知下車地點的紀錄，作為參數'''
    dataset = dataset.drop(check_unknown_do_zone(dataset).index, axis=0)
    dataset = dataset.reset_index(drop=True)
    return dataset


uber = remove_unknown_do_zone(uber)
lyft = remove_unknown_do_zone(lyft)
yellow = remove_unknown_do_zone(yellow)
green = remove_unknown_do_zone(green)

# %% taxi_zones
# %%% 讀入taxi_zones
taxi_zones = pd.read_csv('../taxi_zones.csv')
# 移除Unknown區域
taxi_zones = taxi_zones.drop([263, 264], axis=0)
# 移除不需要的column
taxi_zones = taxi_zones.drop(['Zone', 'service_zone'], axis=1)
# %%% 將taxi_zones的行政區轉換為整數值
# 0 = EWR
# 1 = Queens
# 2 = Bronx
# 3 = Manhattan
# 4 = Staten Island
# 5 = Brooklyn


labels = ['EWR', 'Queens', 'Bronx', 'Manhattan', 'Staten Island', 'Brooklyn']
taxi_zones['borough_id'] = pd.Categorical(
    taxi_zones['Borough'], categories=labels)
taxi_zones['borough_id'] = taxi_zones['borough_id'].cat.codes
taxi_zones = taxi_zones.drop('Borough', axis=1)
del labels

# %% 增加上下車5+1行政區標籤

def merge_with_tz(dataset):
    dataset = dataset.merge(taxi_zones, left_on='PULocationID', right_on='LocationID', how='left')
    dataset = dataset.drop(['LocationID'], axis=1)
    
    dataset = dataset.merge(taxi_zones, left_on='DOLocationID', right_on='LocationID', how='left')
    dataset = dataset.drop(['LocationID'], axis=1)
    
    dataset = dataset.rename(columns={'borough_id_x': 'PUborough_id', 'borough_id_y': 'DOborough_id'})
    dataset[['PUborough_id', 'DOborough_id']].astype('int')
    return dataset


uber = merge_with_tz(uber)
lyft = merge_with_tz(lyft)
yellow = merge_with_tz(yellow)
green = merge_with_tz(green)

# %% 檢查行程時間的極端值

# 雖都佔整體不到1%，但因數量多，不刪除，進行替換
# 考慮到資料量與極右偏分布，用第1到第3四分位數之間的隨機值替換
# 缺點1：沒有考慮到大於0但異常短的行程時間
# 缺點2：狹長型的行政區如曼哈頓可能會多刪一些
# %%% 定義function


def outlier_detect(data, col, attr, threshold=3):
    '''
    data傳入Uber, Lyft, 小黃, 小綠的dataset
    col傳入要處理的column，例如'trip_time(s)'
    attr可傳入：'same', 'diff', 'EWR'
    '''
    # 上下車地點相同，且皆非0 (EWR)的records，IQR倍數設為3
    if attr == 'same':
        process_data = data[(data['PUborough_id'] == data['DOborough_id']) &
                            (data['PUborough_id'] != 0) &
                            (data['DOborough_id'] != 0)]

    # 上下車地點不同，且皆非0 (EWR)的records，IQR倍數設為3
    elif attr == 'diff':
        process_data = data[(data['PUborough_id'] != data['DOborough_id']) &
                            (data['PUborough_id'] != 0) &
                            (data['DOborough_id'] != 0)]

    # 上下車地點其中一個為0 (EWR)的records，IQR倍數設為5 (因EWR距離其他地區較遠)
    elif attr == 'EWR':
        process_data = data[(data['PUborough_id'] == 0) |
                            (data['DOborough_id'] == 0)]
        threshold = 5

    IQR = process_data[col].quantile(0.75) - process_data[col].quantile(0.25)
    Lower_fence = 0
    Upper_fence = process_data[col].quantile(0.75) + (IQR * threshold)
    para = (Upper_fence, Lower_fence)
    tmp = pd.concat([process_data[col] > Upper_fence,
                     process_data[col] < Lower_fence], axis=1)
    outlier_index = tmp.any(axis=1)

    print('Num of outlier detected:', outlier_index.value_counts()[1])
    print(
        f'Proportion of outlier detected: {round(outlier_index.value_counts()[1] / len(outlier_index) * 100, 5)}%')
    print('Upper bound:', para[0])
    print(process_data['trip_time(s)'].describe())
    return process_data, outlier_index, para


def replace_outlier(data, col, process_data, outlier_index, para):
    '''
    data傳入Uber, Lyft, 小黃, 小綠的dataset
    col傳入要處理的column，須與傳入outlier_detect()的一致
    process_data, outlier_index, para為呼叫outlier_detect()回傳的結果
    '''
    # 取得極端值的index，並轉為list
    outlier_index = list(outlier_index[outlier_index == True].index)

    data_copy = data.copy(deep=True)

    # 產生隨機值，範圍在第1到第3四分位數之間，長度與須插捕的數量相同
    random_values = np.random.uniform(process_data[col].quantile(0.25),
                                      process_data[col].quantile(0.75),
                                      len(outlier_index))
    # 將極端值替換為對應位置的隨機值
    for i, j in enumerate(outlier_index):
        data_copy.loc[j, col] = random_values[i]

    return data_copy


# %%% 按「上下車同行政區 --> 上車或下車EWR --> 上下車不同行政區」的順序處理

# 上下車同行政區
for df in [uber, lyft, yellow, green]:
    print('------------------------------')
    process_data, index, para = outlier_detect(df, 'trip_time(s)', 'same')
    df.loc[:, :] = replace_outlier(
        df, 'trip_time(s)', process_data, index, para)


# 上車或下車為EWR
# 小綠無明顯極端值，確認後可不處理
green[(green['PUborough_id'] == 0) | (green['DOborough_id'] == 0)]['trip_time(s)'].describe()
for df in [uber, lyft, yellow]:
    print('------------------------------')
    process_data, index, para = outlier_detect(df, 'trip_time(s)', 'EWR')
    df.loc[:, :] = replace_outlier(
        df, 'trip_time(s)', process_data, index, para)


# 上下車不同行政區
for df in [uber, lyft, yellow, green]:
    print('------------------------------')
    process_data, index, para = outlier_detect(df, 'trip_time(s)', 'diff')
    df.loc[:, :] = replace_outlier(
        df, 'trip_time(s)', process_data, index, para)

del process_data, index, para
# %%% 繪圖查看插捕後分布
for df in [uber, lyft, yellow, green]:
    sns.histplot(df['trip_time(s)'])

plt.figure(figsize=(8, 6))
sns.histplot(uber['trip_time(s)'],
             color='gray', alpha=0.7, label='Uber')
sns.histplot(lyft['trip_time(s)'], color='red', alpha=0.7, label='Lyft')
sns.histplot(yellow['trip_time(s)'],
             color='yellow', alpha=0.7, label='Yellow')
sns.histplot(green['trip_time(s)'],
             color='green', alpha=0.7, label='Green')
plt.title('Trip time (s) of all datasets', fontdict={'fontsize': 18})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Trip time (s)', fontdict={'fontsize': 16})
plt.ylabel('Num of records', fontdict={'fontsize': 16})
plt.legend(fontsize=16)
plt.show()

# %% 組合所有Datasets
#%%% 先標記每一個Dataset的類型
# 0 = Uber
# 1 = Lyft
# 2 = Yellow
# 3 = Green

datasets = {'uber': 0,
            'lyft': 1,
            'yellow': 2,
            'green': 3}

for dataset, value in datasets.items():
    globals()[dataset]['service_type'] = value

del datasets

#%%% 組合
all_data = pd.concat([uber, lyft, yellow, green])
all_data = all_data.sort_values('pickup_datetime').reset_index(drop=True)
all_data.shape

# %% 分出年、月、日、星期、小時欄位

all_data = all_data.assign(year=all_data['pickup_datetime'].dt.year,
                           month=all_data['pickup_datetime'].dt.month,
                           day=all_data['pickup_datetime'].dt.day,
                           weekday=all_data['pickup_datetime'].dt.weekday,
                           hour=all_data['pickup_datetime'].dt.hour)


# %% 節日標籤
# %%% 聯邦法定節日
# import holidays
# ny_holidays = holidays.US(subdiv='NY', years=2020)
# for date, name in sorted(ny_holidays.items()):
#     print(date, name)

# 定義聯邦法定節日，移除非法定節日
class FedHolidays(US):
    def _populate(self, year):
        super()._populate(year)
        self.pop_named("Lincoln's Birthday")
        self.pop_named("Susan B. Anthony Day")


# 實例化FedHolidays
ny_holidays_fed = FedHolidays(subdiv='NY', years=2020)
# 查看
print('---Federal New York Holidays---')
for i in sorted(ny_holidays_fed.items()):
    print(i)
print()

# 定義所有節日


class AllHolidays(FedHolidays):
    def _populate(self, year):
        super()._populate(year)
        self._add_holiday("Valentine's Day", 2, 14)
        self._add_holiday("St.Patrick's Day", 3, 17)
        self._add_holiday("Halloween", 10, 31)
        self._add_holiday("Christmas Eve", 12, 24)


# 實例化AllHolidays
ny_holidays_all = AllHolidays(subdiv='NY', years=2020)
# 查看
print('---All New York Holidays---')
for j in sorted(ny_holidays_all.items()):
    print(j)

del i, j
# %%% 增加聯邦法定節日標籤
all_data['is_fed_holiday'] = np.where(all_data['pickup_datetime'].dt.date.isin(ny_holidays_fed), True, False)
print(all_data['is_fed_holiday'].value_counts())

# %%% 增加所有節日標籤
all_data['is_holiday'] = np.where(all_data['pickup_datetime'].dt.date.isin(ny_holidays_all), True, False)
print(all_data['is_holiday'].value_counts())

# %% 按日期時間計算總量
# 按日期
p_day = all_data.pivot_table(index='day', columns='service_type', aggfunc='size')

# 按小時
p_hour = all_data.pivot_table(index='hour', columns='service_type', aggfunc='size')
p_hour.plot(kind='bar', rot=0, width=1)
plt.legend(fontsize=16)


# %% 測試
