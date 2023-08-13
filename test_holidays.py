# %% Import
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')


def read_data(data_time_1, data_time_2):
    '''
    傳入的data_time格式為"2020-01"、"2022-12"
    '''
    fhvhv_1 = pd.read_parquet(f'../fhvhv_tripdata_{data_time_1}.parquet', engine='pyarrow')
    fhvhv_2 = pd.read_parquet(f'../fhvhv_tripdata_{data_time_2}.parquet', engine='pyarrow')
    fhvhv = pd.concat([fhvhv_1, fhvhv_2]).reset_index(drop=True)
    
    uber = fhvhv[fhvhv.hvfhs_license_num == 'HV0003'].sort_values('pickup_datetime').reset_index(drop=True)
    uber = uber.rename(columns={'trip_time': 'trip_time(s)'})
    uber = uber.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID',
                        'DOLocationID', 'trip_miles', 'trip_time(s)']]
    
    
    lyft = fhvhv[fhvhv.hvfhs_license_num == 'HV0005'].sort_values('pickup_datetime').reset_index(drop=True)
    lyft = lyft.rename(columns={'trip_time': 'trip_time(s)'})
    lyft = lyft.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID',
                        'DOLocationID', 'trip_miles', 'trip_time(s)']]
    
    
    yellow_1 = pd.read_parquet(f'../yellow_tripdata_{data_time_1}.parquet', engine='pyarrow')
    yellow_2 = pd.read_parquet(f'../yellow_tripdata_{data_time_2}.parquet', engine='pyarrow')
    yellow = pd.concat([yellow_1, yellow_2]).sort_values('tpep_pickup_datetime').reset_index(drop=True)
    yellow = yellow.rename(columns={'tpep_pickup_datetime': 'pickup_datetime',
                                    'tpep_dropoff_datetime': 'dropoff_datetime',
                                    'trip_distance': 'trip_miles'})
    yellow = yellow[yellow['pickup_datetime'].dt.to_period('M').between(data_time_1, data_time_2)]
    yellow['trip_time(s)'] = yellow['dropoff_datetime'] - yellow['pickup_datetime']
    yellow['trip_time(s)'] = round(yellow['trip_time(s)'] / np.timedelta64(1, 's'), 0)
    yellow = yellow.loc[:, ['pickup_datetime', 'dropoff_datetime', 'PULocationID', 
                            'DOLocationID', 'trip_miles', 'trip_time(s)']]
    
    
    total_records = fhvhv.shape[0] + lyft.shape[0] + yellow.shape[0]
    print(f'Uber: {uber.shape[0]} {round(uber.shape[0]/total_records*100, 2)}%')
    print(f'Lyft: {lyft.shape[0]} {round(lyft.shape[0]/total_records*100, 2)}%')
    print(f'Yellow: {yellow.shape[0]} {round(yellow.shape[0]/total_records*100, 2)}%')
    print(f'Total: {total_records}')
    
    return uber, lyft, yellow

data_time_1, data_time_2 = '2022-11', '2022-12'

uber, lyft, yellow = read_data(data_time_1, data_time_2)


# %% 檢查缺失值
print('-----Uber-----')
print(uber.info(show_counts=True))
print('\n-----Lyft-----')
print(lyft.info(show_counts=True))
print('\n-----Yellow-----')
print(yellow.info(show_counts=True))

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
taxi_zones['borough_id'] = pd.Categorical(taxi_zones['Borough'], categories=labels)
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


# %% 檢查行程時間的極端值：直接刪除
def outlier_detect(data, col, attr, threshold=3):
    '''
    data傳入Uber, Lyft, 小黃的dataset
    col傳入要處理的column，例如'trip_time(s)'
    attr可傳入：'same', 'diff'
    '''
    # 上下車地點相同
    if attr == 'same':
        process_data = data[data['PUborough_id'] == data['DOborough_id']]

    # 上下車地點不同
    elif attr == 'diff':
        process_data = data[data['PUborough_id'] != data['DOborough_id']]


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


def delete_outlier(data, col, attr, threshold=3):
    '''
    data傳入Uber, Lyft, 小黃的dataset
    col傳入要處理的column，須與傳入outlier_detect()的一致
    process_data, outlier_index, para為呼叫outlier_detect()回傳的結果
    '''
    process_data, outlier_index, _ = outlier_detect(data, col, attr, threshold)
    # 取得極端值的index，並轉為list
    outlier_index = list(outlier_index[outlier_index == True].index)

    data_copy = data.copy(deep=True)
    data_copy = data_copy.drop(outlier_index, axis=0)

    return data_copy


# 刪除，按「上下車同行政區 --> 上車或下車EWR --> 上下車不同行政區」的順序
# 上下車同行政區
print('上下車同行政區')
uber = delete_outlier(uber, 'trip_time(s)', 'same')
lyft = delete_outlier(lyft, 'trip_time(s)', 'same')
yellow = delete_outlier(yellow, 'trip_time(s)', 'same')


# 上下車不同行政區
print('上下車不同行政區')
uber = delete_outlier(uber, 'trip_time(s)', 'diff')
lyft = delete_outlier(lyft, 'trip_time(s)', 'diff')
yellow = delete_outlier(yellow, 'trip_time(s)', 'diff')


# %%% 繪圖查看插捕後分布
# plt.figure(figsize=(8, 6))
# sns.histplot(uber['trip_time(s)'],
#              color='royalblue', alpha=0.7, label='Uber')
# sns.histplot(lyft['trip_time(s)'], color='red', alpha=0.7, label='Lyft')
# sns.histplot(yellow['trip_time(s)'],
#              color='goldenrod', alpha=0.7, label='Yellow')

# plt.title('Trip time (s) of all datasets', fontdict={'fontsize': 18})
# plt.xticks(fontsize=14)
# plt.yticks(fontsize=14)
# plt.xlabel('Trip time (s)', fontdict={'fontsize': 16})
# plt.ylabel('Num of records', fontdict={'fontsize': 16})
# plt.legend(fontsize=16)
# plt.show()

# %% 組合所有Datasets
# 0 = Uber
# 1 = Lyft
# 2 = Yellows

datasets = {'uber': 0,
            'lyft': 1,
            'yellow': 2}

for dataset, value in datasets.items():
    globals()[dataset]['service_type'] = value

del datasets

all_data = pd.concat([uber, lyft, yellow])
all_data = all_data.sort_values('pickup_datetime').reset_index(drop=True)
all_data.shape


# %% 分出年、月、日、星期、小時欄位

all_data = all_data.assign(year=all_data['pickup_datetime'].dt.year,
                           month=all_data['pickup_datetime'].dt.month,
                           day=all_data['pickup_datetime'].dt.day,
                           weekday=all_data['pickup_datetime'].dt.weekday,
                           hour=all_data['pickup_datetime'].dt.hour)


# %% 節日標籤

# 定義節日，移除非法定節日
import datetime
from holidays.countries import US


# from holidays.calendars import gregorian
class AllHolidays(US):
    def _populate(self, year):
        super()._populate(year)
        self.pop_named("Lincoln's Birthday")
        self.pop_named("Susan B. Anthony Day")
        
        self._add_holiday("Valentine's Day", 2, 14)
        self._add_holiday("St.Patrick's Day", 3, 17)
        self._add_holiday("Halloween", 10, 31)
        self._add_holiday("Christmas Eve", 12, 24)
        self._add_holiday("New Year's Eve", 12, 31)


# 實例化AllHolidays
ny_holidays_all = AllHolidays(subdiv='NY', years=[2019, 2020, 2021, 2022, 2023])

# 新增2019~2023年的復活節、復活節星期一
ny_holidays_all[datetime.date(2019, 4, 21)] = "Easter"
ny_holidays_all[datetime.date(2019, 4, 22)] = "Easter Monday"
ny_holidays_all[datetime.date(2020, 4, 12)] = "Easter"
ny_holidays_all[datetime.date(2020, 4, 13)] = "Easter Monday"
ny_holidays_all[datetime.date(2021, 4, 4)] = "Easter"
ny_holidays_all[datetime.date(2021, 4, 5)] = "Easter Monday"
ny_holidays_all[datetime.date(2022, 4, 17)] = "Easter"
ny_holidays_all[datetime.date(2022, 4, 18)] = "Easter Monday"
ny_holidays_all[datetime.date(2023, 4, 9)] = "Easter"
ny_holidays_all[datetime.date(2023, 4, 10)] = "Easter Monday"


# 查看
for i in sorted(ny_holidays_all.items()):
    print(i)

del i

# %%% 增加所有節日標籤
all_data['is_holiday'] = np.where(all_data['pickup_datetime'].dt.date.isin(ny_holidays_all), True, False)
print(all_data['is_holiday'].value_counts())

all_data['is_holiday'].value_counts()

#%% 檢驗節日前夕有沒有差異

p = all_data.pivot_table(index=['month', 'day', 'is_holiday', 'hour'], 
                         columns='PULocationID', 
                         values='trip_miles', 
                         aggfunc='count',
                         fill_value=0)


g = all_data[all_data.month == 12].groupby(['day', 'weekday'])['trip_miles'].size()

def plot_compare_weekday(weekday):
    '''
    weekay參數傳入weekday對應的數值，星期一為0、星期四為3、星期日為6
    '''    
    fig, ax = plt.subplots(figsize=(15, 6))
    bars = ax.bar(range(len(g)), g.values, tick_label=g.index)
    
    ax.set_title('Count of Trips by Day')
    ax.set_xlabel('Day')
    ax.set_ylabel('Count')
    
    x_ticks = range(len(g))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f'{day[0]} ({day[1]})' for day in g.index], rotation=0)
    
    target_weekdays = [weekday]
    for i, bar in enumerate(bars):
        if g.index[i][1] in target_weekdays:
            bar.set_color('red')
    
    plt.tight_layout()
    plt.savefig(f'weekday_{weekday}.png', dpi=300)

plot_compare_weekday(0)
plot_compare_weekday(1)
plot_compare_weekday(2)
plot_compare_weekday(3)
plot_compare_weekday(4)
plot_compare_weekday(5)
plot_compare_weekday(6)


# %% 測試
