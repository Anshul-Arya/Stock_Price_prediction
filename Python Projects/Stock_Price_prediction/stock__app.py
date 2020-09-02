# -*- coding: utf-8 -*-
"""
Created on Thu Sep  3 02:16:06 2020

@author: Anshul Arya
"""

#-------------------------------------------------------#
# Import Libraries
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from dash.dependencies import Input, Output
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
#-------------------------------------------------------#

app = dash.Dash()
server = app.server

scalar = MinMaxScaler(feature_range=(0,1))

# Read the dataset
df_nse = pd.read_csv("file:///C:/Users/Anshul Arya/Desktop/DataScience/" \
                     "Stock-Price-Prediction-Project-Code/" \
                     "NSE-Tata-Global-Beverages-Limited/" \
                     "NSE-Tata-Global-Beverages-Limited.csv")

# Check the data types
df_nse.dtypes

# Convert Date Column to proper Datetime format
df_nse['Date'] = pd.to_datetime(df_nse.Date, format = '%Y-%m-%d')
df_nse.index = df_nse.Date

data = df_nse.sort_index(ascending = True, axis = 0)
new_data = pd.DataFrame(index = range(0, len(df_nse)), columns = ['Date', 'Close'])

for i in range(0, len(data)):
    new_data['Date'][i] = data['Date'][i]
    new_data['Close'][i] = data['Close'][i]
    
new_data.index = new_data.Date
new_data.drop("Date", axis = 1, inplace = True)

dataset = new_data.values

train = dataset[0:987,:]
test = dataset[987:,:]

scalar = MinMaxScaler(feature_range=(0,1))
scaled_data = scalar.fit_transform(dataset)

x_train, y_train = [], []

for i in range(60, len(train)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i,0])
    
x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1],1))

model = load_model("saved_model.h5")

inputs = new_data[len(new_data)-len(test)-60:].values
inputs.reshape(-1,1)
inputs = scalar.transform(inputs)

X_test = []
for i in range(60, inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test,(X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scalar.inverse_transform(closing_price)

train=new_data[:987]
valid=new_data[987:]
valid['Predictions']=closing_price

df = pd.read_csv("file:///C:/Users/Anshul Arya/Desktop/DataScience/" \
                 "Stock-Price-Prediction-Project-Code/stock_data/stock_data.csv")

app.layout = html.Div([
        html.H1("Stock Price Analysis Dashboard", style = {"textAlign":"center"}),
        dcc.Tabs(id = "tabs", children = [
                dcc.Tab(label = 'NSE-TATAGLOBAL Stock Data', children = [
                        html.Div([
                                html.H2("Actual Closing Price", 
                                        style = {"textAlign":"center"}),
                                        dcc.Graph(
                                                id = "Actual Data",
                                                figure = {
                                                        "data":[
                                                                go.Scatter(
                                                                        x = train.index,
                                                                        y = valid.Close,
                                                                        mode = 'markers'
                                                                        )
                                                                ],
                                                        "layout":go.Layout(
                                                                title = "Scatter Plot",
                                                                xaxis = {'title':'Date'},
                                                                yaxis = {'title':'Closing Rate'}
                                                                )
                                                        }),
                                html.H2("LSTM Predicted closing Price", 
                                        style = {"textAlign":"center"}),
                                        dcc.Graph(
                                                id = "Predicted Data",
                                                figure = {
                                                        "data":[
                                                                go.Scatter(
                                                                        x = train.index,
                                                                        y = valid.Predictions,
                                                                        mode = 'markers'
                                                                        )
                                                                ],
                                                        "layout":go.Layout(
                                                                title = "Scatter Plot",
                                                                xaxis = {'title':'Date'},
                                                                yaxis = {'title':'Closing Rate'}
                                                                    )
                                                                }
                                                            )
                                                        
                                ]
                                )                           
                        ]
                        ),
                dcc.Tab(label = 'Facebook Stock Data', children = [
                        html.Div([
                            html.H1("Facebook Stocks High vs Low",
                                    style = {'textAlign':'center'}),
                            dcc.Dropdown(id = 'my-dropdown',
                                         options = [{'label':'Tesla', 'value':'TSLA'},
                                                    {'label':'Apple', 'value':'APPL'},
                                                    {'label':'Facebook', 'value':'FB'},
                                                    {'label':'Microsoft', 'value':'MSFT'},
                                                    ],
                                         multi = True, value = ['FB'],
                                         style = {'display':'block', 'margin-left':'auto',
                                                  'margin-right':'auto', 'width':'60%'}
                                                    ),
                            dcc.Graph(id = 'highlow'),
                            html.H1("Facebook Market Volume", 
                                    style = {'textAlign':'center'}),
                            dcc.Dropdown(id = 'my-dropdown2',
                                         options = [
                                                    {'label':'Tesla', 'value':'TSLA'},
                                                    {'label':'Apple', 'value':'APPL'},
                                                    {'label':'Facebook', 'value':'FB'},
                                                    {'label':'Microsoft', 'value':'MSFT'}
                                                 ],
                                         multi = True, value = ['FB'],
                                         style = {'display':'block', 'margin-left':'auto',
                                                  'margin-right':'auto', 'width':'60%'}
                                         ),
                            dcc.Graph(id = 'volume'),
                                ], className = 'container'),
                    ])                                            
                ])
        ])

@app.callback(Output('highlow', 'figure'),
              [Input('my-dropdown', 'value')])
def update_graph(selected_dropdown):
    dropdown = {'TSLA':'Tesla', 'AAPL':'Apple', 'FB':'Facebook', 'MSFT':'Microsoft',}
    trace1 = []
    trace2 = []
    for stock in selected_dropdown:
        trace1.append(
                go.scatter(x = df[df['Stock'] == stock]["Date"],
                           y = df[df['Stock'] == stock]["High"],
                           mode = 'lines', opacity = 0.7,
                           name = f'High {dropdown[stock]}',
                           textposition = 'bottom center')
            )
        trace2.append(
                go.scatter(x = df[df['Stock'] == stock]["Date"],
                           y = df[df['Stock'] == stock]['Low'],
                           mode = 'lines', opacity = 0.6,
                           name = f'Low {dropdown[stock]}',
                           textposition = 'bottom center')
                )
    traces = [trace1, trace2]
    data = [val for sublist in traces for val in sublist]
    figure = {'data':data,
              'layout':go.Layout(colorway = ["#5E0DAC", '#FF4F00', '#375CB1','#FF7400'
                                             '#FFF400'],
                                 height = 600,
                                 title = f"High and Low prices for {'.'.join(str(dropdown[i]) for i in selected_dropdown)} Over time",
                                 xaxis = {"title":"Date",
                                          'rangeselector':{'buttons':list([{
                                                  'count':1, 'label': '1M',
                                                  'step':'month',
                                                  'stepmode':'backward'},
                                                 {'count':6, 'label': '6M',
                                                  'step':'month', 
                                                  'stepmode':'backward'},
                                                 {'step':'all'}])},
                                          'rangeslider': {'visible':True}, 
                                          'type':'date'},
                                yaxis = {'title':'Price (USD)'})}
    return figure

@app.callback(Output('volume', 'figure'),
              [Input('my-dropdown2','value')])

def update_graph(selected_dropdown_value):
    dropdown = {'TSLA':'Tesla', 'AAPL':'Apple', 'FB':'Facebook', 'MSFT':'Microsoft',}
    trace1 = []
    for stock in selected_dropdown_value:
        trace1.append(
                go.scatter(x = df[df['Stock'] == stock]["Date"],
                           y = df[df['Stock'] == stock]["Volume"],
                           mode = 'lines', opacity = 0.7,
                           name = f'Volume {dropdown[stock]}',
                           textposition = 'bottom center')
            )
    traces = [trace1]
    data = [val for sublist in traces for val in sublist]
    figure = {'data':data,
              'layout':go.Layout(colorway = ["#5E0DAC", '#FF4F00', '#375CB1','#FF7400'
                                             '#FFF400'],
                                 height = 600,
                                 title = f"Market Volume for {'.'.join(str(dropdown[i]) for i in selected_dropdown_value)} Over time",
                                 xaxis = {"title":"Date",
                                          'rangeselector':{'buttons':list([{
                                                  'count':1, 'label': '1M',
                                                  'step':'month',
                                                  'stepmode':'backward'},
                                                 {'count':6, 'label': '6M',
                                                  'step':'month', 
                                                  'stepmode':'backward'},
                                                 {'step':'all'}])},
                                          'rangeslider': {'visible':True}, 
                                          'type':'date'},
                                yaxis = {'title':'Transaction Volume'})}
    return figure
                
if __name__ == '__main__':
    app.run_server(debug=True)
    