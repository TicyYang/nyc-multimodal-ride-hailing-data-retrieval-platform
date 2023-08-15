# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 20:11:50 2023

@author: iSpan
"""
import numpy as np
import pandas as pd
import folium 
from branca.element import Figure

fig3 = Figure(width=1000,height=600)
m3 = folium.Map(location=[40.73439145, -73.95051598], 
                tiles='Stamen Terrain', 
                zoom_start=10, 
                min_zoom=10, 
                max_zoom=16)
fig3.add_child(m3)

# 導入Borough Boundaries
folium.GeoJson('Borough Boundaries.geojson', name='Borough Boundaries').add_to(m3)

m3.save('polygon_map.html')


# %%
ewr = pd.read_csv('../datasets/taxi_zones(from data.gov).csv', usecols=['the_geom'], nrows=1).values

lat = []  # 緯度
lon = []  # 經度


geom_str = ewr[0][0].strip('MULTIPOLYGON ((').strip('))').split()  # 去除頭尾多餘的字串和()，並將每一個值獨立
coordinate = np.array([float(geom_str[-1]), float(geom_str[0])])  # 取出第1個值和最後一個值轉為float，組成一個獨立array，稍後添加到主要array
coordinate = [float(geom_str[-1]), float(geom_str[0])]
# 刪除已經單獨取出的第1個值和最後一個值
geom_str = np.delete(geom_str, 0)
geom_str = np.delete(geom_str, -1)
# 將每個值中多餘的逗號, 空格, ()移除，並轉為float
geom_float = np.array([geom_str])

for j in geom_str:
    geom_float = np.append(geom_float, float(j.strip(', ').strip('((').strip('))')))

geom_float = list(geom_float)

lat_lau_lists = []
for i in range(0, len(geom_float), 2):
    sublist = geom_float[i:i+2]
    lat_lau_lists.append(sublist)

# print(lat_lau_lists)
# print(len(lat_lau_lists))
lat_lau_lists += list(coordinate)
# print(len(lat_lau_lists))
# %%
folium.vector_layers.Polygon(
    locations=lat_lau_lists,
    color='blue',
    fill=True,  # 填充多邊形
    fill_color='green'  # 填充顏色
).add_to(m3)

m3.save('polygon_map.html')



