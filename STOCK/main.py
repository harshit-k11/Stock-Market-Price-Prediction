from analysis import Analysis_page
from pred import stock_pred
import yfinance as yf
import streamlit as st
import datetime
import webbrowser
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



import base64

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
if __name__ == '__main__':
    navi()




