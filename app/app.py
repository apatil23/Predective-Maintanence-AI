import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

# --- Page Config ---
st.set_page_config(
    page_title="Predictive Maintenance AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Load Model ---
model_path = Path("/Users/ajp/pm-ai-env/models/predictive_maintenance_model.pkl")
model = joblib.load(model_path)

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/809/809957.png", width=80)
st.sidebar.title("⚙️ Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "📊 Manual Prediction", "📂 Batch Prediction", "ℹ️ About"])

# --- Home ---
if page == "🏠 Home":
    st.title("🏭 Predictive Maintenance AI Dashboard")
    st.markdown("""
    This tool predicts **machine failure risk** using sensor data.  
    Upload readings or enter values manually to get predictions.
    """)
    st.image(
    "/Users/ajp/pm-ai-env/ai-generated-8949611_1280.jpg",
    caption="AI-Powered Machine Health Monitoring", use_container_width=True
)

# --- Manual Prediction ---
elif page == "📊 Manual Prediction":
    st.header("🔍 Manual Machine Check")
    col1, col2 = st.columns(2)

    with col1:
        air_temperature = st.number_input("🌡️ Air Temperature (K)", value=300.0)
        process_temperature = st.number_input("🔥 Process Temperature (K)", value=312.0)
        rotational_speed = st.number_input("🔄 Rotational Speed (RPM)", value=1500.0)

    with col2:
        torque = st.number_input("⚡ Torque (Nm)", value=40.0)
        tool_wear = st.number_input("⏳ Tool Wear (minutes)", value=120.0)

    if st.button("🚀 Predict"):
        X = pd.DataFrame([{
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear
        }])
        prob = model.predict_proba(X)[:, 1][0]
        pred = int(prob >= 0.5)

        st.metric("Failure Probability", f"{prob:.2%}")
        st.success("✅ No Failure Expected") if pred == 0 else st.error("⚠️ Machine at Risk of Failure!")

# --- Batch Prediction ---
elif page == "📂 Batch Prediction":
    st.header("📂 Upload Sensor Data")
    st.write("Upload a CSV file with columns: `air_temperature, process_temperature, rotational_speed, torque, tool_wear`")

    file = st.file_uploader("Upload CSV", type=["csv"])
    if file is not None:
        df = pd.read_csv("/Users/ajp/pm-ai-env/data/predictive_maintenance_data.csv")
        probs = model.predict_proba(df)[:, 1]
        preds = (probs >= 0.5).astype(int)
        df["failure_probability"] = probs
        df["prediction"] = preds

        st.dataframe(df.head(10))
        st.download_button("⬇️ Download Predictions", df.to_csv(index=False).encode("utf-8"), "predictions.csv", "text/csv")

# --- About ---
elif page == "ℹ️ About":
    st.subheader("👨‍💻 About this Project")
    st.markdown("""
    - **Built with:** Python, Scikit-learn, Streamlit  
    - **Goal:** Predict machine failures before they happen  
    - **Business Value:** Reduce downtime, save maintenance costs  
    """)
    st.info("Created as a demo for Junior AI Engineer role at Co-Ex-Tec.")
