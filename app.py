import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

# Function to determine if a feature is categorical or continuous
def is_categorical(series, threshold=10):
    return series.nunique() <= threshold or series.dtype == 'object'

# Load your dataframe (replace with your actual data loading code)

def extract_numeric(value):
    try:
        # Extract numeric characters and convert to float
        return float(''.join(c for c in str(value) if c.isdigit() or c == '.'))
    except ValueError:
        # Return NaN for non-convertible values
        return float('nan')


@st.cache_data
def load_data():
    # Load the dataset
    data = pd.read_csv('./data/aston_v8vantage_query.csv')
    if 'Mileage' in data.columns:
        # Apply the custom function to clean up mileage values
        data['Mileage'] = data['Mileage'].apply(extract_numeric)
    del data['VIN']
    del data['Displacement']
    return data

data = load_data()

st.title("Vehicle Price Analysis")

# Add tabs for different functionalities
tabs = st.tabs(["Feature Analysis", "Linear Model Analysis", 'Raw Data'])

# Tab 1: Feature Analysis
with tabs[0]:
    st.header("Feature Analysis")

    print(data.columns)
    # Feature selection
    features = [col for col in data.columns if col not in ['Price', 'PriceHistory', 'Unnamed: 0', 'VIN']]
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

# Tab 2: Linear Model Analysis
with tabs[1]:
    st.header("Linear Model Analysis")

    # Select features to include in the model
    available_features = [col for col in data.columns if col not in ['Price', 'PriceHistory', 'Unnamed: 0','Color_Exterior', 'Color_Interior']]
    selected_features = st.multiselect("Select features to include in the model:", available_features, default=available_features)

    if selected_features:
        # Prepare the data
        data_subset = data[selected_features + ['Price']].dropna()
        X = data_subset[selected_features]
        y = data_subset['Price']

        # Handle categorical variables
        X = pd.get_dummies(X, drop_first=True)

        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Get model coefficients
        coefficients = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": model.coef_
        }).sort_values(by="Coefficient", ascending=False)

        # Predict and evaluate the model
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # Display results
        st.subheader("Model Coefficients")
        st.write(coefficients)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error: {mse:.2f}")
        st.write(f"R-squared: {r2:.2f}")

        # Plot predicted vs actual prices
        scatter_fig = px.scatter(
            x=y_test,
            y=y_pred,
            labels={'x': 'Actual Price', 'y': 'Predicted Price'},
            title="Predicted vs Actual Price"
        )
        scatter_fig.update_traces(marker=dict(size=8, opacity=0.7, line=dict(width=1, color='DarkSlateGrey')))
        st.plotly_chart(scatter_fig)

    else:
        st.write("Please select at least one feature to train the model.")

with tabs[2]:
    st.dataframe(data)