import streamlit as st
import pandas as pd
import plotly.express as px

# Function to determine if a feature is categorical or continuous
def is_categorical(series, threshold=10):
    return series.nunique() <= threshold or series.dtype == 'object'

# Load your dataframe (replace with your actual data loading code)
@st.cache_data
def load_data():
    # Example dataframe (replace with actual data loading code)
    data = pd.read_csv('./data/aston_v8vantage_query.csv')
    return data

data = load_data()

st.title("Vehicle Price vs. Features")

# Feature selection
features = [col for col in data.columns if col not in ['Price', 'PriceHistory', 'Unnamed: 0']]
selected_feature = st.selectbox("Select a feature to analyze:", features)

# Determine if the feature is categorical or continuous
if is_categorical(data[selected_feature]):
    # Bar chart for categorical features
    chart_data = data.groupby(selected_feature)['Price'].mean().reset_index()
    fig = px.bar(chart_data, x=selected_feature, y='Price', title=f"Average Price by {selected_feature}")
else:
    # Scatterplot for continuous features
    fig = px.scatter(data, x=selected_feature, y='Price', title=f"Price vs. {selected_feature}")

# Display the chart
st.plotly_chart(fig)
