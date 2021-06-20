

import talib
import ta
import pandas as pd
import requests
from nsepy import get_history
import numpy as np
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model
from datetime import date
import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
yf.pdr_override()
import streamlit as st
import datetime
import base64

import keras.backend.tensorflow_backend as tb
tb._SYMBOLIC_SCOPE.value = True
yf.pdr_override()
today = datetime.date.today()


def navi():
    pages= {
        'Analysis Stock': Analysis_page,
        'Get Predictions': stock_pred
    }
    st.sidebar.title("DASHBOARD")

    selection = st.sidebar.radio("Select your page", tuple(pages.keys()))
    pages[selection]()





main_bg = "14.jpg"
main_bg_ext = "jpg"

side_bg = '16.jpg'
side_bg_ext = "jpg"

st.markdown(
    f"""
    <style>
    .reportview-container {{
        background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
    }}
   .sidebar .sidebar-content {{
        background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
    }}
    </style>
    """,
    unsafe_allow_html=True
)
def Analysis_page():
    st.title("WELCOME TO THE STOCK ANALYSIS")
    st.header('Here you can analysis the stocks for any company by their Symbol')

    main_bg = "14.jpg"
    main_bg_ext = "jpg"

    side_bg = '16.jpg'
    side_bg_ext = "jpg"

    st.markdown(
        f"""
            <style>
            .reportview-container {{
                background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
            }}
           .sidebar .sidebar-content {{
                background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
            }}
            </style>
            """,
        unsafe_allow_html=True
    )

    def user_input_features():
        ticker = st.text_input("Analysis", 'BHEL.NS')
        start_date = st.text_input("Start Date", '2019-01-01')
        end_date = st.text_input("End Date", f'{today}')

        return ticker, start_date, end_date


    symbol, start, end = user_input_features()


    def get_symbol(symbol):
        url = "http://d.yimg.com/autoc.finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
        result = requests.get(url).json()
        for x in result['ResultSet']['Result']:
            if x['symbol'] == symbol:
                return x['name']


    company_name = get_symbol(symbol.upper())
    start = pd.to_datetime(start)
    end = pd.to_datetime(end)
    data = yf.download(symbol, start, end)

    st.header(f"Adjusted Close Price\n {company_name}")
    st.line_chart(data['Adj Close'])
    data['SMA'] = talib.SMA(data['Adj Close'], timeperiod=20)
    data['EMA'] = talib.EMA(data['Adj Close'], timeperiod=20)

    st.header(f"Simple Moving Average vs. Exponential Moving Average\n {company_name}")
    st.line_chart(data[['Adj Close', 'SMA', 'EMA']])
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod=20)
    st.header(f"Bollinger Bands\n {company_name}")
    st.line_chart(data[['Adj Close', 'upper_band', 'middle_band', 'lower_band']])

    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26,
                                                                    signalperiod=9)
    st.header(f"Moving Average Convergence Divergence\n {company_name}")
    st.line_chart(data[['macd', 'macdsignal']])

    cci = ta.trend.cci(data['High'], data['Low'], data['Close'], n=31, c=0.015)
    st.header(f"Commodity Channel Index\n {company_name}")
    st.line_chart(cci)

    data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)
    st.header(f"Relative Strength Index\n {company_name}")
    st.line_chart(data['RSI'])

    data['OBV'] = talib.OBV(data['Adj Close'], data['Volume']) / 10 ** 6
    st.header(f"On Balance Volume\n {company_name}")
    st.line_chart(data['OBV'])


def stock_pred():
    st.title("STOCK PREDICTION WEB APPLICATION")
    st.header('Here you can select the stock you want to predict')
    st.write("This Prediction is done by LSTM model.")


    main_bg = "14.jpg"
    main_bg_ext = "jpg"

    side_bg = '16.jpg'
    side_bg_ext = "jpg"

    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background: url(data:image/{main_bg_ext};base64,{base64.b64encode(open(main_bg, "rb").read()).decode()})
        }}
       .sidebar .sidebar-content {{
            background: url(data:image/{side_bg_ext};base64,{base64.b64encode(open(side_bg, "rb").read()).decode()})
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

    choose_stock = st.selectbox('Choose the stock', ['NONE', 'apple.NS', 'BHEL.NS', 'HDFCBANK.NS', 'TVSMOTOR.NS'])
    if (choose_stock== 'aaple.NS'):
        df1 = get_history(symbol ='TCS',start =date(2020,1,1), end=date.today())
        df1['Date']= df1.index
        st.header('apple')
        if st.checkbox('Show Raw Data'):
            st.subheader("Wanna See Raw Data")
            st.dataframe(df1.tail())

        new_df = df1.filter(['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))

        last_30_days = new_df[-100:].values
        last_30_days_scaled = scaler.fit_transform(last_30_days)
        X_test = []
        X_test.append(last_30_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model = load_model(r'C:\Users\Harshit\Downloads\apple.h5')
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)

        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        get_pred = st.button('GeT Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])

    if(choose_stock=='BHEL.NS'):
        df1 = get_history(symbol='BHEL', start=date(2020, 1, 1), end=date.today())
        df1['Date'] = df1.index
        st.header('BHEL')
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")
            st.dataframe(df1.tail())

        new_df = df1.filter(['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))

        last_30_days = new_df[-100:].values
        last_30_days_scaled = scaler.fit_transform(last_30_days)
        X_test = []
        X_test.append(last_30_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model = load_model('MODELS/BHEL.model')
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)

        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        get_pred = st.button('GeT Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])

    if(choose_stock=='HDFCBANK.NS'):
        df1 = get_history(symbol='HDFCBANK', start=date(2020, 3, 1), end=date.today())
        df1['Date'] = df1.index
        st.header('HDFC')
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")
            st.dataframe(df1.tail())

        new_df = df1.filter(['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))

        last_30_days = new_df[-100:].values
        last_30_days_scaled = scaler.fit_transform(last_30_days)
        X_test = []
        X_test.append(last_30_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model = load_model('MODELS/HDFCBANK.model')
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)

        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        get_pred = st.button('GeT Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])

    if (choose_stock=='TVSMOTOR.NS'):
        df1 = get_history(symbol='TVSMOTOR', start=date(2020, 4, 1), end=date.today())
        df1['Date'] = df1.index
        st.header('TVSMOTOR')
        if st.checkbox('Show Raw Data'):
            st.subheader("Showing raw data---->>>")
            st.dataframe(df1.tail())

        new_df = df1.filter(['Close'])
        scaler = MinMaxScaler(feature_range=(0, 1))

        last_30_days = new_df[-100:].values
        last_30_days_scaled = scaler.fit_transform(last_30_days)
        X_test = []
        X_test.append(last_30_days_scaled)
        X_test = np.array(X_test)
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        model = load_model('MODELS/TVSMOTOR.model')
        pred_price = model.predict(X_test)
        pred_price = scaler.inverse_transform(pred_price)

        NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

        get_pred = st.button('GeT Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])


if __name__ == '__main__':
    navi()




