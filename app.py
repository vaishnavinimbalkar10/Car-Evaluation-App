import streamlit as st
import pandas as pd
import joblib

# ---------------- LOAD MODEL ----------------
model = joblib.load("car_model.pkl")
columns = joblib.load("model_columns.pkl")

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Evaluation App",
    page_icon="🚗",
    layout="centered"
)

# ---------------- TITLE ----------------
st.markdown(
    "<h1 style='text-align: center;'>🚗 Car Evaluation Prediction</h1>",
    unsafe_allow_html=True
)
st.markdown(
    "<p style='text-align: center;'>Predict car acceptability using Machine Learning</p>",
    unsafe_allow_html=True
)
st.divider()

# ---------------- INPUT SECTION ----------------
st.subheader("🔧 Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    buying = st.selectbox("Buying Price", ["low", "med", "high", "vhigh"])
    maint = st.selectbox("Maintenance Cost", ["low", "med", "high", "vhigh"])
    doors = st.selectbox("Number of Doors", ["2", "3", "4", "5more"])

with col2:
    persons = st.selectbox("Seating Capacity", ["2", "4", "more"])
    lug_boot = st.selectbox("Luggage Boot Size", ["small", "med", "big"])
    safety = st.selectbox("Safety Level", ["low", "med", "high"])

# ---------------- PREDICTION ----------------
if st.button("🚀 Predict Car Class"):
    new_car = pd.DataFrame({
        "buying": [buying],
        "maint": [maint],
        "doors": [doors],
        "persons": [persons],
        "lug_boot": [lug_boot],
        "safety": [safety]
    })

    new_car_encoded = pd.get_dummies(new_car)
    new_car_encoded = new_car_encoded.reindex(columns=columns, fill_value=0)

    prediction = model.predict(new_car_encoded)[0]

    st.success(f"✅ Predicted Car Class: **{prediction.upper()}**")

    # Extra interpretation
    if prediction == "unacc":
        st.info("❌ This car is not acceptable.")
    elif prediction == "acc":
        st.info("⚠️ This car is acceptable but average.")
    elif prediction == "good":
        st.info("👍 This car is good.")
    else:
        st.info("🌟 This car is very good!")

# ---------------- FOOTER ----------------
st.markdown("---")
st.caption("Built with ❤️ using Streamlit & Machine Learning")
