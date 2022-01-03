import dash
from dash import html
from dash import dcc
from datetime import date
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split

app = dash.Dash(__name__)
server = app.server
app.layout = html.Div([
    html.Div([
        html.H1("STONKS"),
        html.Div([
            html.P("input stock code:"),
            dcc.Input(id="stock_code", type="text", placeholder=""),
            html.Button('Submit', id='stock_submit')
        ]),
        html.Div([
            html.P("start date:"),
            dcc.DatePickerRange(id='date-picker')
        ]),
        html.Div([
            html.Button("Stock price", id="button1"),
            html.Button("Indicators", id="button2"),
        ]),
        html.Div([
            html.Button('30 Day forecast', id="forecast_submit")
        ])
    ], className="input"),
    html.Div([
        html.Div([], id="header", className="header"),
        html.Div(id="description", className="description_ticker"),
        html.Div([], id="graphs"),
        html.Div([], id="main"),
        html.Div([], id="forecast"),
    ], className="Content")
], className="container")


@app.callback(
    dash.Output(component_id="description", component_property="children")
    , [dash.Input(component_id="stock_submit", component_property="n_clicks")],
    [dash.State(component_id="stock_code", component_property="value")],
    prevent_initial_call=True
)
def update_data(n, val):
    ticker = yf.Ticker(val)
    inf = ticker.info
    df = pd.DataFrame().from_dict(inf, orient="index").T
    output1 = df['longBusinessSummary'].iloc[0]
    return output1


@app.callback(
    dash.Output("graphs", "children"),
    [dash.Input("date-picker", "start_date"),
     dash.Input("date-picker", "end_date"),
     dash.Input("button1", "n_clicks")
     ],
    dash.State("stock_code", "value"),
    prevent_initial_call=True
)
def update_graph(start_date, end_date,  n, stock_code):
    if start_date is None or stock_code is None:
        raise dash.exceptions.PreventUpdate
    else:
        df = yf.download(stock_code, start_date, end_date)
        df.reset_index(inplace=True)
        fig = get_stock_price_fig(df)
        return dcc.Graph(figure=fig)


def get_stock_price_fig(df):
    fig = px.line(df, x='Date', y=['Open', 'Close'], title="Closing and Opening Price vs Date")
    return fig


@app.callback(
    dash.Output("main", "children"),
    [dash.Input("date-picker", "start_date"),
     dash.Input("date-picker", "end_date"),
     dash.Input("button2", "n_clicks")
     ],
    dash.State("stock_code", "value"),
    prevent_initial_call=True
)
def update_indicator(start_date, end_date, n, stock_code):
    if start_date is None or stock_code is None or n is None:
        raise dash.exceptions.PreventUpdate
    else:
        df = yf.download(stock_code, start_date, end_date)
        df.reset_index(inplace=True)
        fig = get_more(df)
        return dcc.Graph(figure=fig)


def get_more(df):
    df['EWA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    fig = px.scatter(df, x='Date', y='EWA_20', title="Exponential Moving Average vs Date")
    fig.update_traces(mode="lines+markers")
    return fig


@app.callback(
    dash.Output("forecast", "children"),
    [dash.Input("date-picker", "end_date"),
     dash.Input("forecast_submit", "n_clicks")],
    dash.State("stock_code", "value"),
    prevent_initial_call=True
)
def update_forecast(end_date, n_clicks, stock_code):

    if stock_code is None:
        raise dash.exceptions.PreventUpdate
    else:

        num_days_forecast = 30
        df = yf.download(stock_code, "2015-01-01", end_date)
        df = df[['Close']]

        df['Prediction'] = df[['Close']].shift(-num_days_forecast)

        x = np.array(df.drop(['Prediction'], 1))
        x = x[:-num_days_forecast]

        y = np.array(df['Prediction'])
        y = y[:-num_days_forecast]

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1)

        lm = LinearRegression()
        lm.fit(x_train, y_train)

        x_forecast = np.array(df.drop(['Prediction'], 1))[-num_days_forecast:]
        x_forecast = np.flip(x_forecast)
        prediction = lm.predict(x_forecast)
        fig = get_forecast_fig(prediction)
        return dcc.Graph(figure=fig)


def get_forecast_fig(forecast):
    fig = px.line(forecast, title="stock forecast")
    return fig


if __name__ == '__main__':
    app.run_server(debug=True)
