import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, mean_absolute_error, mean_squared_error
from imblearn.over_sampling import SMOTE
import plotly.express as px

# Load and preprocess the data
@st.cache_data
def load_data():
    data = pd.read_csv('clean_data.csv')  # Replace with your actual data path or upload method
    data['date'] = pd.to_datetime(data['date'])

    # Feature Engineering: Create necessary columns for model
    data['output_per_hour'] = data['output_hour'] / data['hours']
    data['GDP_per_worker'] = data['gdp'] / data['employment']
    data['log_GDP'] = np.log(data['gdp'] + 1)  # Adding 1 to avoid log(0)
    
    # Calculating key indicators for employment distribution
    data['employed_employer_percentage'] = data['employed_employer'] / data['employment'] * 100
    data['employed_employee_percentage'] = data['employed_employee'] / data['employment'] * 100
    data['employed_own_account_percentage'] = data['employed_own_account'] / data['employment'] * 100
    data['employed_unpaid_family_percentage'] = data['employed_unpaid_family'] / data['employment'] * 100

    # Handle any 'inf' or invalid values by replacing them with NaN
    data.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # Handle NaN values: Fill NaN with column mean for numeric columns
    numeric_data = data.select_dtypes(include=[np.number])  # Only numeric columns
    data[numeric_data.columns] = numeric_data.fillna(numeric_data.mean())  # Fill NaN values in numeric columns

    return data

# Save model using XGBoost (for XGBoost models, JSON format)
def save_xgb_model(model, model_name="trained_xgb_model.json"):
    model.save_model(model_name)

# Load model using XGBoost (for XGBoost models, JSON format)
@st.cache_resource
def load_xgb_model(model_name="trained_xgb_model.json"):
    model = xgb.XGBClassifier()
    model.load_model(model_name)
    return model

# Preprocess data: Scaling, Encoding, etc.
def preprocess_data(data):
    # Label Encoding for sector (convert sector labels to numeric values)
    encoder = LabelEncoder()
    y = data['sector']
    y_encoded = encoder.fit_transform(y)
    
    # Feature engineering and scaling
    X = data[['gdp', 'employment', 'hours', 'output_per_hour', 'GDP_per_worker', 'log_GDP', 
              'employed_employer_percentage', 'employed_employee_percentage', 
              'employed_own_account_percentage', 'employed_unpaid_family_percentage']]
    
    # Train-test split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)
    
    # Replace infinite values with NaN and then handle them (fill with mean)
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_train.fillna(X_train.mean(), inplace=True)

    # Scaling features: Fit the scaler on training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)  # Fit and transform the training data
    
    # Transform the test data with the fitted scaler
    X_test_scaled = scaler.transform(X_test)  # Transform the test data with the fitted scaler

    # Handling imbalanced data using SMOTE
    smote = SMOTE(random_state=42)
    X_train_sm, y_train_sm = smote.fit_resample(X_train_scaled, y_train)
    
    return X_train_sm, X_test_scaled, y_train_sm, y_test, encoder, scaler

# Train models
def train_models(X_train_sm, y_train_sm):
    models = {
        "XGBoost": xgb.XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=3, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    results = []
    for name, model in models.items():
        model.fit(X_train_sm, y_train_sm)
        results.append((name, model))
    
    return results

# Evaluate models and generate results
def evaluate_models(models, X_test, y_test, encoder):
    results = []
    for name, model in models:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        # Metrics
        acc = accuracy
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        results.append({
            "Model": name,
            "Accuracy": acc,
            "MAE": mae,
            "RMSE": rmse,
            "Precision (macro)": report['macro avg']['precision'],
            "Recall (macro)": report['macro avg']['recall'],
            "F1-Score (macro)": report['macro avg']['f1-score'],
        })
    
    return pd.DataFrame(results)

# Employment Distribution by Sector (Pie Chart)
def plot_employment_distribution(data):
    sector_emp_total = data.groupby('sector')['employment'].sum().sort_values(ascending=False)
    fig_pie = px.pie(sector_emp_total, values=sector_emp_total.values, 
                     names=sector_emp_total.index,
                     title="Share of Total Employment by Sector")
    st.plotly_chart(fig_pie)

# Plot GDP Trend Over Time by Sector (Line Plot)
def plot_gdp_trend_by_sector(data):
    plt.figure(figsize=(12, 8))
    sns.lineplot(data=data, x='date', y='gdp', hue='sector', marker='o')
    plt.title('GDP Trend Over Time by Sector')
    plt.xlabel('Date')
    plt.ylabel('GDP')
    plt.xticks(rotation=45)
    plt.grid(True)
    st.pyplot(plt)

# Scatter Plots: GDP vs Other Features
def plot_scatter_gdp_vs_other(data):
    x = data['gdp']
    y_employment = data['employment']
    y_hours = data['hours']
    y_output_hour = data['output_hour']
    y_output_emp = data['output_employment']

    plt.figure(figsize=(20, 5))

    # GDP vs Employment
    plt.subplot(1, 4, 1)
    plt.scatter(x, y_employment, color='steelblue', alpha=0.6, label='Observed Data')
    z = np.polyfit(x, y_employment, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), color='darkred', linestyle='--', linewidth=2, label='Linear Fit')
    plt.xlabel('GDP')
    plt.ylabel('Employment')
    plt.title('GDP vs Employment')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # GDP vs Total Working Hours
    plt.subplot(1, 4, 2)
    plt.scatter(x, y_hours, color='green', alpha=0.6, label='Observed Data')
    z = np.polyfit(x, y_hours, 1)
    p = np.poly1d(z)
    plt.plot(x, p(x), color='darkred', linestyle='--', linewidth=2, label='Linear Fit')
    plt.xlabel('GDP')
    plt.ylabel('Total Working Hours')
    plt.title('GDP vs Total Working Hours')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # GDP vs Output per Hour
    plt.subplot(1, 4, 3)
    plt.scatter(x, y_output_hour, color='orange', alpha=0.6, label='Observed Data')
    plt.xlabel('GDP')
    plt.ylabel('Output per Hour')
    plt.title('GDP vs Output per Hour')
    plt.grid(True, alpha=0.3)

    # GDP vs Output per Employee
    plt.subplot(1, 4, 4)
    plt.scatter(x, y_output_emp, color='purple', alpha=0.6, label='Observed Data')
    plt.xlabel('GDP')
    plt.ylabel('Output per Employee')
    plt.title('GDP vs Output per Employee')
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    st.pyplot(plt)

# Correlation Heatmap
def plot_heatmap(data):
    key_vars = ['gdp', 'employment', 'hours', 'output_hour', 'output_employment']
    corr_matrix = data[key_vars].corr()
    st.write("Correlation Matrix:")
    st.write(corr_matrix)

    plt.figure(figsize=(8,6))
    sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="plasma")
    plt.title("Correlation Heatmap")
    st.pyplot(plt)

# Main function
def main():
    st.title("Interactive Employment Sector Prediction Dashboard")
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 EDA",
        "📈 Trends",
        "🤖 ML Results",
        "🚰 Employment Predictions",
        "ℹ️ About"
    ])
    
    # Load data and preprocess
    data = load_data()
    X_train_sm, X_test, y_train_sm, y_test, encoder, scaler = preprocess_data(data)
    models = train_models(X_train_sm, y_train_sm)
    model_results = evaluate_models(models, X_test, y_test, encoder)
    
    # TAB 1: EDA (Exploratory Data Analysis)
    with tab1:
        st.header("📊 Descriptive Analysis")
        
        # Display Employment Distribution Pie Chart
        plot_employment_distribution(data)
        
        # Display scatter plots for GDP vs other variables
        plot_scatter_gdp_vs_other(data)
        
        # Display the heatmap
        plot_heatmap(data)

    # TAB 2: Trends
    with tab2:
        st.header("📈 GDP Trends Over Time")
        plot_gdp_trend_by_sector(data)

    # TAB 3: ML Results (Model Evaluation)
    with tab3:
        st.header("🤖 Model Performance Results")
        
        # Show model performance comparison table
        st.subheader("Model Performance Comparison")
        st.write(model_results)

    # TAB 4: Employment Predictions
    with tab4:
        st.header("🚰 Predict Employment Sector")

        # User input fields
        gdp_input = st.number_input("💰 Enter GDP (in Billion USD)", min_value=0.0, value=500.0)
        employment_input = st.number_input("👥 Enter Employment", min_value=0, value=500000)
        hours_input = st.number_input("⏰ Enter Total Working Hours", min_value=0, value=40)

        # Button to make prediction
        if st.button("🔮 Predict Employment Sector"):
            # Call the prediction function
            predicted_sector = make_prediction(gdp_input, employment_input, hours_input)

            # Show the result to the user with some enhancements
            st.markdown(f"""
            ### 🎯 **Predicted Employment Sector:**
            #### _{predicted_sector}_

            **Explanation:**
            - This prediction was made based on the economic indicators you provided.
            - The model uses historical data of GDP, employment, and working hours to predict the most likely sector for the given region.

            🔍 **Key Insights**:
            - Employment in sectors like **{predicted_sector}** are influenced by these factors.

            ✨ **Thank you for using our prediction tool!** We hope this helps in your decision-making process.
            """)

            # Option to make another prediction
            st.markdown("#### 🔄 Would you like to make another prediction?")
            if st.button("💬 Make Another Prediction"):
                st.experimental_rerun()

# Run the application
if __name__ == "__main__":
    main()
