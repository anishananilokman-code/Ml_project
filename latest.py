import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ===============================
# PAGE CONFIGURATION
# ===============================
st.set_page_config(
    page_title="Employment Sector Prediction",
    page_icon="🏢",
    layout="wide"
)

# ===============================
# FILE UPLOAD
# ===============================
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

if uploaded_file is not None:
    # Load the uploaded CSV file
    data = pd.read_csv(uploaded_file)

    # Data Preprocessing and Visualization Code...
    st.write(data.head())

    # ===============================
    # Data Preprocessing
    # ===============================
    def preprocess_data(data):
        # Feature Engineering Example:
        data['GDP_per_worker'] = data['gdp'] / data['employment']
        data['output_per_hour'] = data['output_hour'] / data['hours']
        data['log_GDP'] = np.log(data['gdp'] + 1)

        # Label Encoding for sector (assuming sector is categorical)
        encoder = LabelEncoder()
        data['sector'] = encoder.fit_transform(data['sector'])

        # Handle Infinite and NaN Values
        data.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinite values with NaN
        data.fillna(data.mean(), inplace=True)  # Replace NaN values with the mean of each column

        # Split data into features and target
        X = data[['gdp', 'employment', 'hours', 'output_per_hour', 'GDP_per_worker', 'log_GDP']]
        y = data['sector']

        # Split into training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        return X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler

    X_train_scaled, X_test_scaled, y_train, y_test, encoder, scaler = preprocess_data(data)

    # ===============================
    # NAVIGATION (Sidebar)
    # ===============================
    st.sidebar.title("Navigation")
    tab = st.sidebar.radio("Go to", ["Project Overview & Motivation", "Data Overview", "📊 EDA", "📈 Trends", "📊 Sector Productivity", "📉 Employment Distribution", "💼 Sector GDP Trend", "📊 Model Overview", "📝 System Information"])

    # ===============================
    # TAB 1: Project Overview & Motivation
    # ===============================
    if tab == "Project Overview & Motivation":
        st.markdown("<h1>Project Overview & Motivation</h1>", unsafe_allow_html=True)

        st.subheader("1. Problem Statement")
        st.markdown("""
        The imbalance in the structure of labour force participation and productivity performance between key economic sectors remains, despite Malaysia’s effort to continually upgrade labour market efficiency and sectoral productivity. Certain key sectors have shown relatively high employment participation with low productivity, while others contribute significantly to national output with fewer workers.
        """)

        st.subheader("2. Motivation of Project")
        st.markdown("""
        Different sectors contribute to Malaysian economic growth through a variety of means. While GDP can measure the performance of various sectors economically, a more accurate indicator of productivity is how well labour resources are used. This research project seeks to examine in detail the labour force status and productivity level by sector to help policymakers ensure strength in workforce allocation, productivity improvement, and sustainable economic growth.
        """)

        st.subheader("3. Project Objectives")
        st.markdown("""
        1. To develop a machine learning model that classifies employment sectors based on productivity indicators, working hours, labour force size, and GDP contribution.
        2. To train and compare multiple classification models to identify the best performing algorithm.
        3. To deploy the final model in the dashboard.
        """)

        st.subheader("4. Project Limitations")
        st.markdown("""
        Limitations of the current research include the reliance on secondary data from the **Department of Statistics Malaysia (DOSM)**, which may have some inconsistencies. Additionally, the accuracy of predictions depends on the available indicators and may vary over time.
        """)

    # ===============================
    # TAB 2: Data Overview
    # ===============================
    if tab == "Data Overview":
        st.markdown("<h1>Data Overview</h1>", unsafe_allow_html=True)
        st.subheader("Data Sources")

        # Display Data Sources and Description
        st.markdown("""
        - **Department of Statistics Malaysia (DOSM)** provided secondary data used in this study.
        - The dataset includes information on employment, productivity, GDP, working hours, and other key variables.
        """)

        st.subheader("Data Structure and Features")
        st.markdown("""
        **Features**:
        - **Date**: Year of the data.
        - **Sector**: Employment sector.
        - **GDP**: Gross domestic product.
        - **Hours Worked**: Total hours worked.
        - **Employment**: Number of employed persons in the sector.
        - **Output per Hour**: Ratio of GDP to hours worked.
        - **Output Employment**: Ratio of GDP to the number of employed persons.
        """)

        st.write(data.head())

    # ===============================
    # TAB 3: EDA (Exploratory Data Analysis)
    # ===============================
    if tab == "📊 EDA":
        st.markdown("<h1>Exploratory Data Analysis (EDA)</h1>", unsafe_allow_html=True)
        st.subheader("Correlation Matrix")

        # Key variables for correlation
        key_vars = ['gdp', 'employment', 'hours', 'output_hour', 'output_employment']

        # Correlation matrix
        corr_matrix = data[key_vars].corr()
        fig_corr = plt.figure(figsize=(8,6))
        sns.heatmap(corr_matrix,
