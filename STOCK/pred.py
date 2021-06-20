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

    choose_stock = st.selectbox('Choose the stock', ['NONE','BHEL', 'HDFCBANK', 'TVSMOTOR'])
    # if (choose_stock== 'TCS'):
    #     df1 = get_history(symbol ='apple',start =date(2020,1,1), end=date.today())
    #     df1['Date']= df1.index
    #     st.header('TCS')
    #     if st.checkbox('Show Raw Data'):
    #         st.subheader("Want to see Raw Data")
    #         st.dataframe(df1.tail())

    #     new_df = df1.filter(['Close'])
    #     scaler = MinMaxScaler(feature_range=(0, 1))

    #     last_30_days = new_df[-100:].values
    #     last_30_days_scaled = scaler.fit_transform(last_30_days)
    #     X_test = []
    #     X_test.append(last_30_days_scaled)
    #     X_test = np.array(X_test)
    #     X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    #     model = load_model(r'MODELS/TCS.model')
    #     pred_price = model.predict(X_test)
    #     pred_price = scaler.inverse_transform(pred_price)

    #     NextDay_Date = datetime.date.today() + datetime.timedelta(days=1)

    #     get_pred = st.button('Get Prediction of Selected Stock')
    #     if get_pred:
    #         st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
    #         st.markdown(pred_price)

    #     st.subheader("Close Price VS Date Interactive chart for analysis:")
    #     st.line_chart(df1['Close'])

    #     st.subheader("Line chart of Open and Close for analysis:")
    #     st.line_chart(df1[['Open', 'Close']])

    #     st.subheader("Line chart of High and Low for analysis:")
    #     st.line_chart(df1[['High', 'Low']])

    if(choose_stock=='BHEL'):
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

        get_pred = st.button('Get Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])

    if(choose_stock=='HDFCBANK'):
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

        get_pred = st.button('Get Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])

    if (choose_stock=='TVSMOTOR'):
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

        get_pred = st.button('Get Prediction of Selected Stock')
        if get_pred:
            st.subheader("Predictions for the next upcoming day Close Price : " + str(NextDay_Date))
            st.markdown(pred_price)

        st.subheader("Close Price VS Date Interactive chart for analysis:")
        st.line_chart(df1['Close'])

        st.subheader("Line chart of Open and Close for analysis:")
        st.line_chart(df1[['Open', 'Close']])

        st.subheader("Line chart of High and Low for analysis:")
        st.line_chart(df1[['High', 'Low']])


