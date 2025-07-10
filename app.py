
# Note: Ensure that streamlit is installed in your environment using:
# pip install -r requirements.txt

try:
    import streamlit as st
except ModuleNotFoundError:
    raise ModuleNotFoundError("Streamlit is not installed. Please run 'pip install streamlit' before executing this app.")

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data and model
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_csv("input_model_potenc_predXfault7_A.csv")

    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df.drop(columns=[col for col in df.columns if df[col].isnull().mean() > 0.4], inplace=True)
    df.fillna(df.median(numeric_only=True), inplace=True)
    df['hour'] = df['timestamp'].dt.hour
    df['dayofweek'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df.drop(columns=['timestamp'], inplace=True)
    return df

@st.cache_resource
def train_model(df):
    X = df.drop(columns=['fault_d7'])
    y = df['fault_d7']
    X_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    model.fit(X_train, y_train)
    return model, X.columns.tolist()

# Main Streamlit App
st.title("🚰 Urban Water Leak Prediction")
st.markdown("Predict potential pipeline faults using ML and sensor data")

# File upload
uploaded_file = st.sidebar.file_uploader("Upload your water sensor CSV file", type=["csv"])

# Load data and train model
with st.spinner("Loading data and training model..."):
    data = load_data(uploaded_file)
    model, feature_list = train_model(data)

# User Input
st.sidebar.header("Sensor Input Features")
user_input = {}
for feature in feature_list:
    user_input[feature] = st.sidebar.slider(
        feature, float(data[feature].min()), float(data[feature].max()), float(data[feature].median())
    )

input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict Fault"):
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.error(f"⚠️ Fault Predicted! Probability: {prob:.2f}")
    else:
        st.success(f"🛠️ No Fault Detected. Probability: {prob:.2f}")

    st.subheader("Feature Contribution")
    importances = model.feature_importances_
    importance_df = pd.DataFrame({"Feature": feature_list, "Importance": importances})
    importance_df = importance_df.sort_values(by="Importance", ascending=False).head(10)
    st.bar_chart(importance_df.set_index("Feature"))

# Footer
st.markdown("---")
st.caption("Built for Urban Infrastructure Monitoring")
