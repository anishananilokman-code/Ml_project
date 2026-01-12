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
# BACKGROUND & STYLING
# ===============================
def apply_custom_styles():
    st.markdown(
        """
        <style>
        .stApp {
            background-color: #f3f4f6;  /* Light background color */
            color: black; /* Black text color */
        }
        .stButton>button {
            background-color: #0073e6;
            color: white;
            font-weight: bold;
            border-radius: 10px;
            box-shadow: 0 2px 6px rgba(0, 0, 0, 0.2);
            transition: background-color 0.3s ease;
        }
        .stButton>button:hover {
            background-color: #005bb5;
        }
        .block-container {
            background-color: rgba(255,255,255,0.92);
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .stTextInput>label {
            font-size: 1rem;
        }
        .stSelectbox, .stMultiselect {
            background-color: #ffffff;
            border-radius: 5px;
        }
        .stMetric>label {
            font-size: 1.2rem;
        }
        .stSidebar {
            background-color: #2b2d42;
        }
        .stSidebar .sidebar-content a {
            color: #fff;
        }
        .stTitle, .stSubheader {
            color: #0073e6;
            font-family: 'Arial', sans-serif;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

apply_custom_styles()

# ===============================
# MAIN TITLE
# ===============================
st.markdown("<h1 style='color: #0073e6;'>🏢 Employment Sector Prediction Dashboard</h1>", unsafe_allow_html=True)
st.subheader("Predicting Employment Sectors based on Economic Indicators")
st.caption("📊 GDP | 🏭 Productivity | 💼 Work Hours | 👥 Labor Force")

# ===============================
# FILE UPLOADER
# ===============================
uploaded_file = st.file_uploader("Upload CSV file", type="csv")

# Check if the user uploaded a file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)  # Read the uploaded file

    # Display the first few rows of the uploaded file
    st.write(data.head())

    # Proceed with preprocessing and model
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

    # Example Model Evaluation (XGBoost)
    model = xgb.XGBClassifier(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    # Calculate Accuracy, Precision, Recall, and F1-score
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    # Display the metrics
    st.subheader("Model Performance")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Precision: {precision:.2f}")
    st.write(f"Recall: {recall:.2f}")
    st.write(f"F1-Score: {f1:.2f}")

    # Model comparison (if multiple models are trained)
    model_comparison = pd.DataFrame({
        'Model': ['XGBoost'],
        'Accuracy': [accuracy],
        'Precision': [precision],
        'Recall': [recall],
        'F1-Score': [f1]
    })
    st.write(model_comparison)

    # ===============================
    # VISUALIZE EDA (Correlation Heatmap)
    # ===============================
    st.subheader("Correlation Matrix")
    key_vars = ['gdp', 'employment', 'hours', 'output_hour', 'output_employment']
    corr_matrix = data[key_vars].corr()
    fig_corr = plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="plasma")
    plt.title("Correlation Heatmap")
    st.pyplot(fig_corr)

    # ===============================
    # PREDICTION (Interactive Prediction)
    # ===============================
    st.subheader("Prediction")
    gdp_input = st.sidebar.number_input("GDP (Billion USD)", min_value=0.0, max_value=5000.0, value=1000.0, step=100.0)
    work_hours_input = st.sidebar.number_input("Average Work Hours", min_value=0, max_value=100, value=40, step=1)
    employment_input = st.sidebar.number_input("Employment Figures", min_value=0, max_value=1000000, value=500000, step=5000)
    output_per_hour_input = st.sidebar.number_input("Output per Hour (units)", min_value=0.0, max_value=1000.0, value=150.0, step=10.0)

    gdp_per_worker_input = gdp_input / employment_input if employment_input != 0 else 0
    log_gdp_input = np.log(gdp_input + 1)

    input_data = np.array([[gdp_input, employment_input, work_hours_input, output_per_hour_input, gdp_per_worker_input, log_gdp_input]])
    input_data_scaled = scaler.transform(input_data)

    if st.sidebar.button("Predict Sector", key="predict_button"):
        prediction = model.predict(input_data_scaled)
        predicted_sector = encoder.inverse_transform(prediction)[0]

        st.markdown(f"### 🎯 Predicted Employment Sector: {predicted_sector}")

        # Visualize the prediction
        st.subheader("Visualizing the Entered Data")
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        data_for_plot = pd.DataFrame({
            'Input Feature': ['GDP', 'Employment', 'Work Hours', 'Output per Hour'],
            'Value': [gdp_input, employment_input, work_hours_input, output_per_hour_input]
        })
        ax.bar(data_for_plot['Input Feature'], data_for_plot['Value'], color=['#3498db', '#2ecc71', '#e74c3c', '#9b59b6'])
        ax.set_title("Entered Economic Indicators", fontsize=16)
        ax.set_ylabel("Value")
        st.pyplot(fig)
else:
    st.write("Please upload a CSV file to proceed.")
