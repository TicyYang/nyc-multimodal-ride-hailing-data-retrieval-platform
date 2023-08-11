# %% Import
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



del fhvhv_202011, fhvhv_202012, fhvhv, yellow_202011, yellow_202012
# %% 查看各Dataset紀錄數
total_records = uber.shape[0] + lyft.shape[0] + yellow.shape[0]

print(f'Uber: {uber.shape[0]} {round(uber.shape[0]/total_records*100, 2)}%')
print(f'Lyft: {lyft.shape[0]} {round(lyft.shape[0]/total_records*100, 2)}%')
print(f'Yellow: {yellow.shape[0]} {round(yellow.shape[0]/total_records*100, 2)}%')
print(f'Total: {total_records}')
# Uber: 16861697 64.99%
# Lyft: 6114098 23.56%
# Yellow: 2970898 11.45%
# Total: 25946693

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

# %% 欄位改名
# Uber、Lyft行程時間、車資欄位改名
uber = uber.rename(columns={'trip_time': 'trip_time(s)',
                                          'base_passenger_fare': 'base_fare'})
lyft = lyft.rename(columns={'trip_time': 'trip_time(s)',
                                          'base_passenger_fare': 'base_fare'})

# 小黃上車時間、下車時間、行駛距離、車資欄位改名
yellow = yellow.rename(columns={'tpep_pickup_datetime': 'pickup_datetime',
                                              'tpep_dropoff_datetime': 'dropoff_datetime',
                                              'trip_distance': 'trip_miles',
                                              'fare_amount': 'base_fare'})

# %% 檢查缺失值
print('-----Uber-----')
print(uber.info(show_counts=True))
print('\n-----Lyft-----')
print(lyft.info(show_counts=True))
print('\n-----Yellow-----')
print(yellow.info(show_counts=True))

# %% 檢查小黃的日期時間
# yellow_wrong_time = yellow[yellow['pickup_datetime'].dt.to_period('M') != '2020-11']
# yellow_wrong_time = yellow_wrong_time.sort_values(by='pickup_datetime').reset_index(drop=True)
# print(f'Num of yellow wrong pickup datetime: {yellow_wrong_time.shape[0]}')

# yellow_wrong_time_1031 = yellow_wrong_time[yellow_wrong_time['pickup_datetime'].dt.to_period('D') == '2020-10-31']
# print(f'Num of yellow wrong pickup datetime at 2020-10-31: {yellow_wrong_time_1031.shape[0]}')
# yellow_wrong_time_1201 = yellow_wrong_time[yellow_wrong_time['pickup_datetime'].dt.to_period('D') == '2020-12-01']
# print(f'Num of yellow wrong pickup datetime at 2020-12-01: {yellow_wrong_time_1201.shape[0]}')
# 112筆中有48筆為10月31日的資料，應可歸於10月的資料，於此月分析先剔除，與所有資料結合時可保留
# 112筆中有16筆為12月1日的資料，應可歸於12月的資料，於此月分析先剔除，與所有資料結合時可保留
# 其餘48筆日期差距較大，應予刪除
yellow = yellow[yellow['pickup_datetime'].dt.to_period('M').between('2020-11', '2020-12')]
# del yellow_wrong_time, yellow_wrong_time_1031, yellow_wrong_time_1201


# %% 計算小黃的行程時間
yellow['trip_time(s)'] = yellow['dropoff_datetime'] - yellow['pickup_datetime']
yellow['trip_time(s)'] = round(yellow['trip_time(s)'] / np.timedelta64(1, 's'), 0)

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


# %% 檢查行程時間的極端值

# 考慮到資料量與極右偏分布，用第1到第3四分位數之間的隨機值替換
# 缺點1：沒有考慮到大於0但異常短的行程時間
# 缺點2：狹長型的行政區如曼哈頓可能會多刪一些
# %%% 定義function
def outlier_detect(data, col, attr, threshold=3):
    '''
    data傳入Uber, Lyft, 小黃的dataset
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
    data傳入Uber, Lyft, 小黃的dataset
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
print('上下車同行政區')
for df in [uber, lyft, yellow]:
    print('------------------------------')
    process_data, index, para = outlier_detect(df, 'trip_time(s)', 'same')
    df.loc[:, :] = replace_outlier(
        df, 'trip_time(s)', process_data, index, para)


# 上車或下車為EWR
print('上車或下車為EWR')
for df in [uber, lyft, yellow]:
    print('------------------------------')
    process_data, index, para = outlier_detect(df, 'trip_time(s)', 'EWR')
    df.loc[:, :] = replace_outlier(
        df, 'trip_time(s)', process_data, index, para)


# 上下車不同行政區
print('上下車不同行政區')
for df in [uber, lyft, yellow]:
    print('------------------------------')
    process_data, index, para = outlier_detect(df, 'trip_time(s)', 'diff')
    df.loc[:, :] = replace_outlier(
        df, 'trip_time(s)', process_data, index, para)

del process_data, index, para

# %%% 另一種：直接刪除
# 簡化版的outlier_detect
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
plt.figure(figsize=(8, 6))
sns.histplot(uber['trip_time(s)'],
             color='royalblue', alpha=0.7, label='Uber')
sns.histplot(lyft['trip_time(s)'], color='red', alpha=0.7, label='Lyft')
sns.histplot(yellow['trip_time(s)'],
             color='yellow', alpha=0.7, label='goldenrod')

plt.title('Trip time (s) of all datasets', fontdict={'fontsize': 18})
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlabel('Trip time (s)', fontdict={'fontsize': 16})
plt.ylabel('Num of records', fontdict={'fontsize': 16})
plt.legend(fontsize=16)
plt.show()

# %% 組合所有Datasets
# 先標記每一個Dataset的類型
# 0 = Uber
# 1 = Lyft
# 2 = Yellow

datasets = {'uber': 0,
            'lyft': 1,
            'yellow': 2}

for dataset, value in datasets.items():
    globals()[dataset]['service_type'] = value

del datasets

# 組合
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
# %% 假日標籤
# 是聯邦法定節日或週六日，就標為假日
# all_data['is_day_off'] = np.where(
#     (all_data['is_fed_holiday']) | 
#     (all_data['weekday'].isin([5, 6])), 
#     True, False
#     )

# print(all_data['is_day_off'].value_counts())

#%% # 檢驗節日前夕有沒有差異



# %% 按日期時間計算總量


p = all_data.pivot_table(index=['month', 'day', 'is_holiday', 'hour'], 
                         columns='PULocationID', 
                         values='trip_miles', 
                         aggfunc='count',
                         fill_value=0)
# p.loc[(11, 1, 0):(11, 4, 5)]


g = all_data[all_data.month == 12].groupby(['day', 'weekday'])['trip_miles'].size()

def plot_compare_weekday(weekday):
    '''
    weekay參數傳入weekday對應的數值，星期一為0、星期四為3、星期日為6
    '''    
    fig, ax = plt.subplots()
    bars = ax.bar(range(len(g)), g.values, tick_label=g.index)
    
    ax.set_title('Count of Trips by Day in December')
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
    plt.show()


plot_compare_weekday(3)
plot_compare_weekday(4)
plot_compare_weekday(5)
plot_compare_weekday(6)
plot_compare_weekday(0)
plot_compare_weekday(1)
plot_compare_weekday(2)



p = all_data.pivot_table(index=['month', 'day', 'weekday', 'is_holiday', 'hour'], 
                         columns='PULocationID', 
                         values='trip_miles', 
                         aggfunc='count',
                         fill_value=0)

df_p = p.reset_index()
df_p['hour'] = range(len(df_p))
df_p['is_holiday'] = np.where(df_p['is_holiday'] == True, 1, 0)


# X_train = df_p.loc[:49, ['day', 'weekday', 'is_holiday']]
# X_test = df_p.loc[50:, ['day', 'weekday', 'is_holiday']]
# y_train = df_p.loc[:49, 1]
# y_test = df_p.loc[50:, 1]


X = df_p.loc[:, ['hour', 'weekday', 'is_holiday']]
y = df_p[3]

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)




from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score, KFold

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.fit_transform(X_test)


models = {'LinearRegression': LinearRegression(),
          'Decision Tree': DecisionTreeRegressor(),
          'Random Forest': RandomForestRegressor(),
          'SVR': SVR(kernel='rbf', C=1)
          }
results = []

for model in models.values():
    kf = KFold(n_splits=6, random_state=42, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kf)
    results.append(cv_results)

plt.boxplot(results, labels=models.keys())
plt.show()


#%%% 線性迴歸
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LinearRegression()
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r_square = model.score(X_test, y_test)
print(f'R square: {model.score(X_test, y_test)}')

#%% 決策樹
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = DecisionTreeRegressor()
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r_square = model.score(X_test, y_test)
print(f'R square: {model.score(X_test, y_test)}')


#%% 隨機森林
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = RandomForestRegressor()
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r_square = model.score(X_test, y_test)
print(f'R square: {model.score(X_test, y_test)}')

#%% SVR
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = SVR(kernel='rbf', C=1)
model.fit(X_train, y_train)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r_square = model.score(X_test, y_test)
print(f'R square: {model.score(X_test, y_test)}')

#%%% XGBoost
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = XGBRegressor(n_estimators=500, 
                     learning_rate=0.05, 
                     n_jobs=-1, 
                     random_state=0)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=2)

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r_square = model.score(X_test, y_test)
print(f'R square: {model.score(X_test, y_test)}')

#%% LightGBM
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

model = LGBMRegressor(n_estimators=500, 
                        learning_rate=0.1, 
                        n_jobs=-1, 
                        random_state=0)

model.fit(X_train, y_train, eval_set=[(X_test, y_test)])

X_test = scaler.transform(X_test)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
r_square = model.score(X_test, y_test)
print(f'R square: {model.score(X_test, y_test)}')


#%% 
g_11_12 = all_data.groupby(['month', 'day', 'PULocationID'])['trip_miles'].sum()
df = g_11_12.to_frame()


a = df.reset_index()
a['day'] = pd.Categorical(a['day'], ordered=True)
dummy_a = pd.get_dummies(a.PULocationID)
a_dummy_a = pd.concat([a, dummy_a], axis=1) 


dummy_df = pd.get_dummies(df.index.get_level_values('PULocationID'))
result_df = pd.concat([df, dummy_df], axis=1)








p_month_day_hour = all_data.pivot_table(index=['month', 'day', 'hour'], columns='service_type', aggfunc='size')



# %% 測試
