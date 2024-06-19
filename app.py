# Import Necessary Libraries
import pandas as pd
import streamlit as st
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.express as px
import pickle
import plotly.graph_objects as go
import matplotlib.pyplot as plt

# Import Data
data = pd.read_csv(r"C:\Users\Raavin\OneDrive\Desktop\PROJECT 6\Data.unknown")
data.drop(columns=['Unnamed: 0'], inplace=True)
Forecast_df = pd.read_csv(r"C:\Users\Raavin\OneDrive\Desktop\PROJECT 6\Forecasts.unknown")
Forecast_df.drop(columns=['Unnamed: 0'], inplace=True)

# Load the saved model
with open(r"C:\Users\Raavin\OneDrive\Desktop\PROJECT 6\Model.pkl", 'rb') as model_file:
    loaded_model = pickle.load(model_file)

Forecast_df = pd.DataFrame({'Date': pd.date_range(start='2012-10-30', end='2023-01-31')})
Forecast_df = Forecast_df.rename(columns={'Date': 'ds'})

# Streamlit app
st.set_page_config(page_title="Oil Price Forecast", page_icon=":chart_with_upwards_trend:")

# Sidebar with tabs
tab = st.sidebar.radio("Navigation", ["Home", "Data", "Prediction"])

# Home tab
if tab == "Home":
    st.sidebar.header("Home")
    st.title("Oil Price Forecast")
    st.image("https://media.istockphoto.com/id/1216766385/photo/oil-prices-moving-up.jpg?s=612x612&w=0&k=20&c=lx4KuRvns5a_4xdEuHh44TEZYEHQw5fcrL1xcDDoFuQ=", width=850)
    
# Data tab
elif tab == "Data":
    st.sidebar.header("Data")
    st.title("Oil Price Data")
    st.header("Original Data")
    if st.checkbox("Show Data"):
       st.write(data)

    # Time Series plot
    st.header("Time Series Plot")
    if st.checkbox("Show Graph"):
       fig = px.line(data, x='Date', y='Close', title='Time Series Plot of Oil Prices')
       st.plotly_chart(fig)

# Prediction tab
elif tab == "Prediction":
    st.sidebar.header("Prediction")
    st.title("Oil Price Prediction")
    
    # Input for selecting a date
    selected_date = st.date_input(
        "Select a date:",
        value=Forecast_df['ds'].min(),
        min_value=Forecast_df['ds'].min(),
        max_value=Forecast_df['ds'].max()
    )

    # Make prediction for the selected date
    selected_date_df = pd.DataFrame({'ds': [selected_date]})
    forecast = loaded_model.predict(selected_date_df)

    # Display forecasted price at the selected date
    st.subheader("Forecast of Oil Price on Selected Date:")
    st.markdown(f"<p style='font-family: New Times Roman; font-size: 20px;'>Price at {selected_date}: {forecast.loc[0, 'yhat']}</p>", unsafe_allow_html=True)

    # Plot entire time series using plot_plotly
    fig = plot_plotly(loaded_model, forecast)
    st.plotly_chart(fig)