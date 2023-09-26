from flask import Flask, render_template, request
import pandas as pd
import plotly.graph_objs as go
from datetime import datetime
import predict.load_model as model
import json  #新增
from datetime import timedelta
from dateutil.relativedelta import relativedelta

app = Flask(__name__)

# 即時預測開關 關0  開1
Instant_predict_switch = 0

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/team')
def team():
    return render_template('team.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == "GET":
        return render_template('predict.html')
    elif request.method == "POST":
        # # (新增)获取用户选择的日期
        # selected_date = datetime.strptime(request.form['date'], '%Y/%m/%d')

        # # (新增)获取前一天日期
        # previous_date = selected_date - relativedelta(days=1)
        
        
        # 當日
        today = datetime.now().date().strftime('%Y/%m/%d')
        
        # names = ['lyft', 'uber', 'yellow']
        names = ['yellow','lyft', 'uber']
        hours = range(24)
        year = datetime.strptime(request.form['date'], '%Y/%m/%d').year
        month = datetime.strptime(request.form['date'], '%Y/%m/%d').month
        day = datetime.strptime(request.form['date'], '%Y/%m/%d').day
        PULocationID = int(request.form['PULocationID'])
        
        date = datetime(year, month, day) 
        year = date.year
        month = date.month
        day = date.day
            
        json_file_path = "./static/NYC Taxi Zones.geojson" 
        with open(json_file_path, "r") as json_file:
            data_dict = json.load(json_file)
        
        for feature in data_dict["features"]:
            location_id = int(feature["properties"]["location_id"])
            if PULocationID == location_id:
                matched_zone = feature["properties"]["zone"]
                matched_borough = feature["properties"]["borough"]
                break  # 找到匹配后退出循环

        if Instant_predict_switch == 0:                  
            predict_file_path = "./predict/predictions.csv" 
            predict_df = pd.read_csv(predict_file_path)
            #predict_df = pd.read_parquet(predict_file_path)       
            condition = ((predict_df['hour'] >= 0) & (predict_df['hour'] <= 24) & 
                        (predict_df['year'] == year) & 
                        (predict_df['month'] == month) & 
                        (predict_df['day'] == day) & 
                        (predict_df['PULocationID'] == PULocationID))
            filtered_data = predict_df[condition]
            sorted_data = filtered_data.sort_values(by=['Name', 'hour'], ascending=[True, True])
            predict = sorted_data['prediction_integer'].tolist()
        else :
            predict = model.predict_demand(year,month,day,PULocationID)
        
        # 創建日期時間範圍
        datetime_range = pd.date_range(start=f"{year}-{month}-{day}", periods=len(hours), freq='H')

        # 分割預測值列表
        split_predict = [predict[i:i+24] for i in range(0, len(predict), 24)]

        # 創建一个空的DataFrame
        df = pd.DataFrame({'datetime': datetime_range})

        # 添加'names'欄
        for i, name in enumerate(names):
            df[name] = split_predict[i]

        # df.to_csv('./templates/test.csv')
        # df = pd.read_csv('./templates/test.csv')

        # 創建一個包含多個線條的Plotly圖表
        fig = go.Figure()

        # 添加三個線條到圖表
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['lyft'], mode='lines', name='Lyft', line=dict(color='blue')))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['uber'], mode='lines', name='Uber', line=dict(color='green')))
        fig.add_trace(go.Scatter(x=df['datetime'], y=df['yellow'], mode='lines', name='Yellow', line=dict(color='red')))

        # 設置圖表佈局
        fig.update_layout(
            # title=' LocationID：'+ (request.form['PULocationID']),
            # title='Lyft vs Uber vs Yellow',
            xaxis=dict(title='Time'),
            yaxis=dict(title='Demand'),
            margin=dict(t=50, l=0),
        )

        # 將 Plotly 圖表轉換為 HTML 字符串
        graph_html = fig.to_html(full_html=False)

        # (新增)在返回结果页面之前，将前一天的日期传递给结果页面
        # previous_date_str = previous_date.strftime('%Y/%m/%d')

        return render_template('predict_result.html', page_header="Predict Demand", name = names, year = year, month = month, day = day, hour = hours, PULocationID = PULocationID, predict = predict, graph_html=graph_html,matched_zone=matched_zone,matched_borough=matched_borough,today=today,date=date)

@app.route('/previous', methods=['POST'])
def before():
    # 當日
    today = datetime.now().date().strftime('%Y/%m/%d')

    names = ['yellow','lyft', 'uber']
    hours = range(24)
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    PULocationID = int(request.form['PULocationID'])

    date = datetime(year, month, day) - timedelta(days=1)

    year = date.year
    month = date.month
    day = date.day

    json_file_path = "./static/NYC Taxi Zones.geojson" 
    with open(json_file_path, "r") as json_file:
        data_dict = json.load(json_file)
    
    for feature in data_dict["features"]:
        location_id = int(feature["properties"]["location_id"])
        if PULocationID == location_id:
            matched_zone = feature["properties"]["zone"]
            matched_borough = feature["properties"]["borough"]
            break  # 找到匹配后退出循环

    if Instant_predict_switch == 0:      
        # predict_file_path = "./predict/TS10-test.parquet" 
        # predict_df = pd.read_parquet(predict_file_path)  
        predict_file_path = "./predict/predictions.csv" 
        predict_df = pd.read_csv(predict_file_path)
             
        condition = ((predict_df['hour'] >= 0) & (predict_df['hour'] <= 24) & 
                    (predict_df['year'] == year) & 
                    (predict_df['month'] == month) & 
                    (predict_df['day'] == day) & 
                    (predict_df['PULocationID'] == PULocationID))
        filtered_data = predict_df[condition]
        sorted_data = filtered_data.sort_values(by=['Name', 'hour'], ascending=[True, True])
        predict = sorted_data['prediction_integer'].tolist()
    else :
        predict = model.predict_demand(year,month,day,PULocationID)

    
    # 創建日期時間範圍
    datetime_range = pd.date_range(start=f"{year}-{month}-{day}", periods=len(hours), freq='H')

    # 分割預測值列表
    split_predict = [predict[i:i+24] for i in range(0, len(predict), 24)]

    # 創建一个空的DataFrame
    df = pd.DataFrame({'datetime': datetime_range})

    # 添加'names'欄
    for i, name in enumerate(names):
        df[name] = split_predict[i]

    # df.to_csv('./templates/test.csv')
    # df = pd.read_csv('./templates/test.csv')

    # 創建一個包含多個線條的Plotly圖表
    fig = go.Figure()

    # 添加三個線條到圖表
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['lyft'], mode='lines', name='Lyft', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['uber'], mode='lines', name='Uber', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['yellow'], mode='lines', name='Yellow', line=dict(color='red')))

    # 設置圖表佈局
    fig.update_layout(
        # title=' LocationID：'+ (request.form['PULocationID']),
        # title='Lyft vs Uber vs Yellow',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Demand'),
        margin=dict(t=50, l=0),
    )

    # 將 Plotly 圖表轉換為 HTML 字符串
    graph_html = fig.to_html(full_html=False)

    return render_template('predict_result.html', page_header="Predict Demand", name = names, year = year, month = month, day = day, hour = hours, PULocationID = PULocationID, predict = predict, graph_html=graph_html,matched_zone=matched_zone,matched_borough=matched_borough,today=today,date=date)


@app.route('/next', methods=['POST'])
def next():
    # 當日
    today = datetime.now().date().strftime('%Y/%m/%d')
    
    names = ['yellow','lyft', 'uber']
    hours = range(24)
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    PULocationID = int(request.form['PULocationID'])

    date = datetime(year, month, day) + timedelta(days=1)

    year = date.year
    month = date.month
    day = date.day
    
    
    json_file_path = "./static/NYC Taxi Zones.geojson" 
    with open(json_file_path, "r") as json_file:
        data_dict = json.load(json_file)
    
    for feature in data_dict["features"]:
        location_id = int(feature["properties"]["location_id"])
        if PULocationID == location_id:
            matched_zone = feature["properties"]["zone"]
            matched_borough = feature["properties"]["borough"]
            break  # 找到匹配后退出循环


    if Instant_predict_switch == 0:      
        predict_file_path = "./predict/predictions.csv" 
        predict_df = pd.read_csv(predict_file_path)
        # predict_file_path = "./predict/TS10-test.parquet" 
        # predict_df = pd.read_parquet(predict_file_path)       
        condition = ((predict_df['hour'] >= 0) & (predict_df['hour'] <= 24) & 
                    (predict_df['year'] == year) & 
                    (predict_df['month'] == month) & 
                    (predict_df['day'] == day) & 
                    (predict_df['PULocationID'] == PULocationID))
        filtered_data = predict_df[condition]
        sorted_data = filtered_data.sort_values(by=['Name', 'hour'], ascending=[True, True])
        predict = sorted_data['prediction_integer'].tolist()
    else :
        predict = model.predict_demand(year,month,day,PULocationID)

    # 創建日期時間範圍
    datetime_range = pd.date_range(start=f"{year}-{month}-{day}", periods=len(hours), freq='H')

    # 分割預測值列表
    split_predict = [predict[i:i+24] for i in range(0, len(predict), 24)]

    # 創建一个空的DataFrame
    df = pd.DataFrame({'datetime': datetime_range})

    # 添加'names'欄
    for i, name in enumerate(names):
        df[name] = split_predict[i]

    # df.to_csv('./templates/test.csv')
    # df = pd.read_csv('./templates/test.csv')

    # 創建一個包含多個線條的Plotly圖表
    fig = go.Figure()

    # 添加三個線條到圖表
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['lyft'], mode='lines', name='Lyft', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['uber'], mode='lines', name='Uber', line=dict(color='green')))
    fig.add_trace(go.Scatter(x=df['datetime'], y=df['yellow'], mode='lines', name='Yellow', line=dict(color='red')))

    # 設置圖表佈局
    fig.update_layout(
        # title=' LocationID：'+ (request.form['PULocationID']),
        # title='Lyft vs Uber vs Yellow',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Demand'),
        margin=dict(t=50, l=0),
    )

    # 將 Plotly 圖表轉換為 HTML 字符串
    graph_html = fig.to_html(full_html=False)

    return render_template('predict_result.html', page_header="Predict Demand", name = names, year = year, month = month, day = day, hour = hours, PULocationID = PULocationID, predict = predict, graph_html=graph_html,matched_zone=matched_zone,matched_borough=matched_borough,today=today,date=date)

@app.route('/dashboard')
def dashboard():
    return dash_app.index()

import dash
import dash_bootstrap_components as dbc
from dash import Input, Output, html, dcc
import calendar

dash_app = dash.Dash(__name__, server=app, external_stylesheets=[dbc.themes.BOOTSTRAP])

df_hour = pd.read_parquet('TS20-dash-hour.parquet')
df_day = pd.read_parquet('TS21-dash-day.parquet')
df_month = pd.read_parquet('TS22-dash-month.parquet')

navbar = dbc.NavbarSimple(
    children=[
        dbc.DropdownMenu(
            children=[
                dbc.DropdownMenuItem("HOME", href='/', external_link=True),
                dbc.DropdownMenuItem("DASHBOARD", href='/dashboard', external_link=True),
                dbc.DropdownMenuItem("PREDICTION", href='/predict', external_link=True),
                dbc.DropdownMenuItem("TEAM", href='/team', external_link=True)
            ],
            nav=True,
            in_navbar=True,
            label="More"
        )
    ],
    brand="多元乘車服務大數據檢索平台",
    color="dark",
    dark=True
)

sidebar = html.Div(
    [
        html.H4("Filter", className="mt-3"),
        html.P(
            "Name:",
            className="card-text"
        ),
        dcc.Checklist(
            id='Checklist-selector-Name',
            options=[
                {'label': 'Lyft', 'value': 'lyft'},
                {'label': 'Uber', 'value': 'uber'},
                {'label': 'Yellow', 'value': 'yellow'}
            ],
            value=['lyft', 'uber', 'yellow'],  # 預設選擇值為'all'
            labelStyle={'display': 'inline-block',
                        'margin-right': '10%'}
        ),
        html.P(
            "Year:",
            className="card-text"
        ),
        dcc.RangeSlider(
            id='range-slider-selector-Year',
            min=2019,
            max=2023,
            step=1,
            value=[2022, 2023],  
            marks={i: str(i) for i in range(2019, 2024)}  # 將年份以字串形式顯示
        ),
        html.P(
            "Borough:",
            className="card-text"
        ),
        dcc.Dropdown(
            id='dropdown-selector-Borough',
            options=['all', 'Bronx', 'Brooklyn', 'EWR', 'Manhattan', 'Queens', 'Staten Island'],
            value='all'  # 預設全部
        ),
        html.P(
            "PULocationID:",
            className="card-text"
        ),
        dcc.Dropdown(
            id='dropdown-selector-PULocationID'
        )
    ]
)

# 新增連結及改顏色字
header = html.Div(
    html.A(
        html.H2("Dashboard"),
        style={'color': 'black', 'text-decoration': 'none'}
    ),
    className="mt-3"
)

tabs = dbc.Tabs(
    [
        dbc.Tab(
            dcc.Graph(
                id='line-chart',
            ), 
            label="Month"
        ),
        dbc.Tab(
            dcc.Graph(
                id='line-chart-day',
            ), 
            label="Day"
        ),
        dbc.Tab(
            dcc.Graph(
                id='line-chart-hour',
            ), 
            label="Hour"
        )
    ]
)

cards = html.Div(
    [
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Total Ridership", className="card-text"),
                                html.H5(
                                    id='total-rides', 
                                    className="card-title"
                                )
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 4, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 4, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 4, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Average Daily Ridership", className="card-text"),
                                html.H5(
                                    id='average-daily-rides', 
                                    className="card-title"
                                )
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 4, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 4, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 4, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.P("Growth Rate", className="card-text"),
                                html.H5(
                                    id='growth-rate', 
                                    className="card-title"
                                )
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 4, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 4, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 4, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                )
            ],
            className="mb-3"
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                tabs
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 8, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 8, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 8, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            dcc.Graph(
                                                id='pie-chart',
                                            ), 
                                            label="Pie"
                                        )
                                    ]
                                )
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 4, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 4, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 4, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                )
            ],
            className="mb-3"
        ),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dbc.Tabs(
                                    [
                                        dbc.Tab(
                                            dcc.Graph(
                                                id='bar-chart',
                                            ), 
                                            label="Bar"
                                        )
                                    ]
                                )                                
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 6, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 6, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 6, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                dcc.Dropdown(
                                        id='dropdown-selector-number',
                                        options=list(range(10,265,25)),
                                        value=10,  # 預設值
                                        style={'max-width':'350px','width': '50%'}
                                ),
                                dcc.Graph(
                                    id='bar-chart-id',
                                )
                            ]
                        ),
                        className="shadow"
                    ),
                    width={"size": 6, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 6, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 6, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                )
            ],
            className="mb-3"
        )
    ]
)

carousel = dbc.Carousel(
    items=[
        {
            "key": "1",
            "src": "assets/images/manhattan.jpg"
        },
        {
            "key": "2",
            "src": "assets/images/brooklyn.jpg"
        },
        {
            "key": "3",
            "src": "assets/images/queens.jpg"
        },
        {
            "key": "4",
            "src": "assets/images/bronx.jpg"
        },
        {
            "key": "5",
            "src": "assets/images/staten_island.jpg"
        }
    ],
    controls=True,
    indicators=True,
    interval=False,
    variant="dark",
    id="carousel"
)

modal = html.Div(
    [
        dbc.Modal(
            [
                dbc.ModalHeader("預覽圖"),
                dbc.ModalBody(
                    [
                        html.Div(id="modal-image")
                    ]
                )
            ],
            id="modal",
            centered=True
        )
    ]
)

dash_app.layout = html.Div(
    [
        navbar,
        dbc.Row(
            [
                dbc.Col(
                    [
                        sidebar,
                        html.Br(),
                        carousel
                    ], 
                    width={"size": 2, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 2, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 2, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                ),
                dbc.Col(
                    [
                        dcc.Interval(id="interval", interval=500, n_intervals=0),
                        header,
                        html.Br(),
                        cards,
                        html.Br(),
                        modal,
                    ],
                    width={"size": 10, "order": 1, "offset": 0},  # 調整 Col 的寬度和順序
                    lg={"size": 10, "order": 1, "offset": 0},     # 在較大螢幕上保持相同配置
                    md={"size": 10, "order": 1, "offset": 0},     # 在中型螢幕上保持相同配置
                    sm={"size": 11.5, "order": 1, "offset": 0},    # 在較小螢幕上調整 Col 的寬度
                    xs={"size": 11.5, "order": 1, "offset": 0}     # 在最小螢幕上調整 Col 的寬度
                )
            ],
            className="p-3"
        )
    ],
    style={'backgroundColor': '#F0F0F0'}
)

# 判斷是否開啟預覽圖
@dash_app.callback(
    [Output("modal", "is_open"), Output("modal-image", "children")],
    [Input("carousel", "active_index")],
)
def toggle_modal(active_index):
    if active_index is not None:
        selected_item = carousel.items[active_index]
        src = selected_item["src"]
        preview_image = html.Img(src=src, style={"width": "100%"})
        return True, preview_image

    return False, None

# 更新 PULocationID 內容
@dash_app.callback(
    Output('dropdown-selector-PULocationID', 'options'),
    [Input('dropdown-selector-Borough', 'value')]
)
def update_input(borough):

    if borough == 'all':
        options = [
            {'label': f'{id} : {zone}', 'value': id}
            for id, zone in zip(df_month['PULocationID'].unique(), df_month['zone'].unique())
        ]
        return options
    else:
        df_filtered = df_month[df_month['borough'] == borough]
        options = [
            {'label': f'{id} : {zone}', 'value': id}
            for id, zone in zip(df_filtered['PULocationID'].unique(), df_filtered['zone'].unique())
        ]
        return options

# 定義搭乘量單位
def format_total(total):
    if total >= 100000000:
        return f'{total/100000000:.2f} 億'
    elif total >= 10000:
        return f'{total/10000:.2f} 萬'
    else:
        return f'{total:.2f}'

# 更新總搭乘量
@dash_app.callback(
    Output('total-rides', 'children'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_total_rides(name, year, borough, locationid):
    if locationid:
        df_filtered = df_month[(df_month['PULocationID']==locationid) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    else:
        if borough == 'all':
            df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
        else:
            df_filtered = df_month[(df_month['borough'] == borough) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]

    if len(name) == 3:
        total=df_filtered['total'].sum()
    else:    
        total=0
        for i in name:
            total = total+df_filtered[i].sum()

    formatted_total = format_total(total)
    return formatted_total

def days_in_year(year):
    # 判斷是否為閏年
    is_leap = calendar.isleap(year)

    # 獲取每個月的天數
    days_in_month = [31, 29 if is_leap else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    # 計算總天數
    if year == 2019:
        total_days = sum(days_in_month)-31
    elif year == 2023:
        total_days = sum(days_in_month)-184
    else:
        total_days = sum(days_in_month)

    return total_days

# 更新日平均搭乘量
@dash_app.callback(
    Output('average-daily-rides', 'children'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_rides_ratio(name, year, borough, locationid):
    if locationid:
        df_filtered = df_month[(df_month['PULocationID']==locationid) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    else:
        if borough == 'all':
            df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
        else:
            df_filtered = df_month[(df_month['borough'] == borough) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]

    # 搭乘量
    if len(name) == 3:
        total=df_filtered['total'].sum()
    else:    
        total=0
        for i in name:
            total = total+df_filtered[i].sum()

    # 天數
    years = range(year[0], year[1]+1)
    total_days=0
    for y in years:
        total_days = total_days+days_in_year(y)

    average = total/total_days
    formatted_average = format_total(average)
    return formatted_average

# 更新增長率
@dash_app.callback(
    Output('growth-rate', 'children'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_rides_ratio(name, year, borough, locationid):
    if locationid:
        df_filtered = df_month[(df_month['PULocationID']==locationid) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    else:
        if borough == 'all':
            df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
        else:
            df_filtered = df_month[(df_month['borough'] == borough) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]

    if year[1] == 2023:
        df_before = df_filtered[(df_filtered['year'] == year[0]) & (df_filtered['month'] <= 6)]
    else:
        df_before = df_filtered[df_filtered['year'] == year[0]]
    
    if year[0] == 2019:
        df_after = df_filtered[(df_filtered['year'] == year[1]) & (df_filtered['month'] >= 2)]
    else:
        df_after = df_filtered[df_filtered['year'] == year[1]]
    

    if len(name) == 3:
        before = df_before['total'].sum()
        after = df_after['total'].sum()
    else:    
        before = 0
        after=0
        for i in name:
            before = before+df_before[i].sum()
            after = after+df_after[i].sum()

    if before == 0 :
        rate = 0
    else : 
        rate = (after-before)/before

    return f'{rate:.2%}'

# 取得當月最大天數
def get_max_day(row):
    year = row['year']
    month = row['month']
    _, max_day = calendar.monthrange(year, month)
    return max_day

# 更新折線圖(月)
@dash_app.callback(
    Output('line-chart', 'figure'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_plot(name, year, borough, locationid):
    
    if locationid:
        df_filtered = df_month[(df_month['PULocationID']==locationid) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    else:
        if borough == 'all':
            df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
        else:
            df_filtered = df_month[(df_month['borough'] == borough) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    
    # 設定初始的 x 軸範圍
    if year[0] == 2019 :       
        start_date = str(year[0]) + '-02'
    else: 
        start_date = str(year[0]) + '-01'
    if year[1] == 2023 :       
        end_date = str(year[1]) + '-06'
    else: 
        end_date = str(year[1]) + '-12'  
         
    xaxis_range = [start_date, end_date] 
    xaxis_tickvals = [start_date]
    
    if year[1]-year[0] > 1 :    
        while datetime.strptime(start_date, '%Y-%m') + relativedelta(months=6) <= datetime.strptime(end_date, '%Y-%m'):
            start_date = (datetime.strptime(start_date, '%Y-%m')+ relativedelta(months=6)).strftime('%Y-%m')   
            xaxis_tickvals.append(start_date)
    else :
        while datetime.strptime(start_date, '%Y-%m') + relativedelta(months=3) <= datetime.strptime(end_date, '%Y-%m'):
            start_date = (datetime.strptime(start_date, '%Y-%m')+ relativedelta(months=3)).strftime('%Y-%m')   
            xaxis_tickvals.append(start_date)        
        
    xaxis_tickvals.append(end_date)
    xaxis_ticktext = xaxis_tickvals
    
    fig = {
        'data': [],
        'layout': {
            'title': f'Ridership from <span style="color: blue">{year[0]} to {year[1]}</span>',
            'xaxis': {'range': xaxis_range, 'tickvals': xaxis_tickvals, 'ticktext': xaxis_ticktext},  # 設定 x 軸的範圍
            }
    }

    df_filtered = df_filtered.drop(['day', 'datetime'], axis=1)

    df_filtered['day'] = df_filtered.apply(get_max_day, axis=1)

    df_filtered['datetime'] = pd.to_datetime(df_filtered[['year', 'month', 'day']])

    for n in name:
        df_grouped = df_filtered.groupby('datetime')[n].sum().reset_index()
        fig['data'].append({'x': df_grouped['datetime'], 'y': df_grouped[n], 'type': 'line', 'mode': 'lines+markers', 'name': n})

    df_grouped = df_filtered.groupby('datetime')['total'].sum().reset_index()
    fig['data'].append({'x': df_grouped['datetime'], 'y': df_grouped['total'], 'type': 'line', 'mode': 'lines+markers', 'name': 'total', 'line': {'color': 'black'}, 'marker': {'color': 'black'}})

    return fig

# 更新圓餅圖
@dash_app.callback(
    Output('pie-chart', 'figure'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_plot(name, year, borough, locationid):
    if locationid:
        df_filtered = df_month[(df_month['PULocationID']==locationid) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    else:
        if borough == 'all':
            df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
        else:
            df_filtered = df_month[(df_month['borough'] == borough) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    
    values = []
    for n in name:
        values.append(df_filtered[n].sum())

    colors = ['rgb(31, 119, 180)', 'rgb(255, 127, 14)', 'rgb(44, 160, 44)']
    fig = go.Figure(data=[go.Pie(labels=name, values=values, marker=dict(colors=colors))])
    fig.update_layout(title_text=f'Ridership from <span style="color: blue">{year[0]} to {year[1]}</span>',
                      title_x=0.5)

    return fig

# 更新折線圖(日)
@dash_app.callback(
    Output('line-chart-day', 'figure'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_plot(name, year, borough, locationid):
    if locationid:
        df_filtered = df_day[(df_day['PULocationID']==locationid) & (df_day['year'] == year[1])]
    else:
        if borough == 'all':
            df_filtered = df_day[df_day['year'] == year[1]]
        else:
            df_filtered = df_day[(df_day['borough'] == borough) & (df_day['year'] == year[1])]

    df_filtered = df_filtered[df_filtered['month'] == df_filtered['month'].max()].reset_index()
    month = df_filtered['datetime'].iloc[0].strftime('%B')

    # 設定初始的 x 軸範圍
    start_date = str(year[1]) + '-' + str(df_filtered['month'].max()).zfill(2) + '-1'
    _, max_day = calendar.monthrange(year[1], df_filtered['month'].max())
    end_date = str(year[1]) + '-' + str(df_filtered['month'].max()).zfill(2) + '-'+str(max_day)
    xaxis_range = [start_date, end_date] 
    
    xaxis_tickvals = [start_date,
                     (datetime.strptime(start_date, '%Y-%m-%d')+ timedelta(days=7)).strftime('%Y-%m-%d') ,
                     (datetime.strptime(start_date, '%Y-%m-%d')+ timedelta(days=14)).strftime('%Y-%m-%d') ,
                     (datetime.strptime(start_date, '%Y-%m-%d')+ timedelta(days=21)).strftime('%Y-%m-%d') ,
                     end_date]
    #xaxis_ticktext = [start_date, (datetime.strptime(start_date, '%Y-%m-%d')+ timedelta(days=7)).strftime('%Y-%m-%d') , end_date]
    xaxis_ticktext = xaxis_tickvals
    
    fig = {
        'data': [],
        'layout': {
            'title': f'Ridership in <span style="color: blue">{month}  {year[1]}</span>',
            'xaxis': {'range': xaxis_range, 'tickvals': xaxis_tickvals, 'ticktext': xaxis_ticktext},  # 設定 x 軸的範圍
            }
    }

    for n in name:
        df_grouped = df_filtered.groupby('datetime')[n].sum().reset_index()
        fig['data'].append({'x': df_grouped['datetime'], 'y': df_grouped[n], 'type': 'line', 'mode': 'lines+markers', 'name': n})

    df_grouped = df_filtered.groupby('datetime')['total'].sum().reset_index()
    fig['data'].append({'x': df_grouped['datetime'], 'y': df_grouped['total'], 'type': 'line', 'mode': 'lines+markers', 'name': 'total', 'line': {'color': 'black'}, 'marker': {'color': 'black'}})

    return fig

# 更新折線圖(時)
@dash_app.callback(
    Output('line-chart-hour', 'figure'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-PULocationID', 'value')]
)
def update_plot(name, year, borough, locationid):
    if locationid:
        df_filtered = df_hour[(df_hour['PULocationID']==locationid) & (df_hour['year'] == year[1])]
    else:
        if borough == 'all':
            df_filtered = df_hour[df_hour['year'] == year[1]]
        else:
            df_filtered = df_hour[(df_day['borough'] == borough) & (df_hour['year'] == year[1])]

    df_filtered = df_filtered[df_filtered['month'] == df_filtered['month'].max()]
    day = df_filtered['day'].max()
    df_filtered = df_filtered[df_filtered['day'].isin(list(range(day-2,day+1)))].reset_index()
    month = df_filtered['datetime'].iloc[0].strftime('%B')

    fig = {
        'data': [],
        'layout': {'title': f'Ridership from <span style="color: blue">{month} {day-2} to {day}, {year[1]}</span>'}
    }

    for n in name:
        df_grouped = df_filtered.groupby('datetime')[n].sum().reset_index()
        fig['data'].append({'x': df_grouped['datetime'], 'y': df_grouped[n], 'type': 'line', 'mode': 'lines+markers', 'name': n})

    df_grouped = df_filtered.groupby('datetime')['total'].sum().reset_index()
    fig['data'].append({'x': df_grouped['datetime'], 'y': df_grouped['total'], 'type': 'line', 'mode': 'lines+markers', 'name': 'total', 'line': {'color': 'black'}, 'marker': {'color': 'black'}})

    return fig

# 更新長條圖
@dash_app.callback(
    Output('bar-chart', 'figure'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value')]
)
def update_plot(name, year):

    df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    
    if len(name) == 3:
        df_grouped = df_filtered.groupby('borough')['total'].sum().reset_index()
    else:
        df_grouped = df_filtered.groupby('borough')[name].sum().reset_index()
        df_grouped['total'] = 0
        for i in name:
            df_grouped['total'] = df_grouped['total'] + df_grouped[i]

    # 將 df_grouped 按 'total' 欄位的值由大到小排序
    df_grouped = df_grouped.sort_values(by='total', ascending=True)

    fig = go.Figure(data=[go.Bar(
        y=df_grouped['borough'],
        x=df_grouped['total'],
        orientation='h'  # 設置為水平方向
    )])
    fig.update_layout(title_text=f'Borough Ridership Rankings <br> from <span style="color: blue">{year[0]} to {year[1]}</span>',
                      title_x=0.5)

    return fig

# 更新長條圖(zone)
@dash_app.callback(
    Output('bar-chart-id', 'figure'),
    [Input('Checklist-selector-Name', 'value'),
     Input('range-slider-selector-Year', 'value'),
     Input('dropdown-selector-Borough', 'value'),
     Input('dropdown-selector-number', 'value')]
)
def update_plot(name, year, borough, number):

    if borough == 'all':
        df_filtered = df_month[(df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    else:
        df_filtered = df_month[(df_month['borough'] == borough) & (df_month['year'] >= year[0]) & (df_month['year'] <= year[1])]
    
    if len(name) == 3:
        df_grouped = df_filtered.groupby('zone')['total'].sum().reset_index()
    else:
        df_grouped = df_filtered.groupby('zone')[name].sum().reset_index()
        df_grouped['total'] = 0
        for i in name:
            df_grouped['total'] = df_grouped['total'] + df_grouped[i]

    # 將 df_grouped 按 'total' 欄位的值由大到小排序
    df_grouped = df_grouped.sort_values(by='total', ascending=True)

    # 只保留前10筆資料
    df_grouped = df_grouped.tail(number)

    fig = go.Figure(data=[go.Bar(
        y=df_grouped['zone'],
        x=df_grouped['total'],
        orientation='h'  # 設置為水平方向
    )])
    fig.update_layout(title_text=f'Zone Ridership Rankings <br> from <span style="color: blue">{year[0]} to {year[1]}</span>',
                      title_x=0.5)

    return fig


if __name__ == "__main__":
    dash_app.run_server(debug=True)