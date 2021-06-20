import yfinance as yf
import streamlit as st
import talib
import ta
import pandas as pd
import requests
import datetime
import base64
today = datetime.date.today()



def Analysis_page():


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
    st.title("WELCOME TO THE STOCK ANALYSIS")
    
    st.header('Here you can analysis the stocks for any company by their Symbol')
    
    
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
    data = yf.download(symbol,start,end)




    st.header(f"Adjusted Close Price\n {company_name}")
    st.line_chart(data['Adj Close'])
    data['SMA'] = talib.SMA(data['Adj Close'], timeperiod = 20)
    data['EMA'] = talib.EMA(data['Adj Close'], timeperiod = 20)

    st.header(f"Simple Moving Average vs. Exponential Moving Average\n {company_name}")
    st.line_chart(data[['Adj Close','SMA','EMA']])
    data['upper_band'], data['middle_band'], data['lower_band'] = talib.BBANDS(data['Adj Close'], timeperiod =20)
    st.header(f"Bollinger Bands\n {company_name}")
    st.line_chart(data[['Adj Close','upper_band','middle_band','lower_band']])

    data['macd'], data['macdsignal'], data['macdhist'] = talib.MACD(data['Adj Close'], fastperiod=12, slowperiod=26, signalperiod=9)
    st.header(f"Moving Average Convergence Divergence\n {company_name}")
    st.line_chart(data[['macd','macdsignal']])

    cci= ta.trend.cci(data['High'], data['Low'], data['Close'], n=31, c=0.015)
    st.header(f"Commodity Channel Index\n {company_name}")
    st.line_chart(cci)

    data['RSI'] = talib.RSI(data['Adj Close'], timeperiod=14)
    st.header(f"Relative Strength Index\n {company_name}")
    st.line_chart(data['RSI'])

    data['OBV'] = talib.OBV(data['Adj Close'], data['Volume'])/10**6
    st.header(f"On Balance Volume\n {company_name}")
    st.line_chart(data['OBV'])


