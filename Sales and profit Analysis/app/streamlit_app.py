import streamlit as st
import numpy as np
from src.save_model import load

model = load()

st.set_page_config(
    page_title="Store Sales AI",
    layout="wide"
)

st.title("Store Sales Prediction Dashboard")

st.sidebar.header("Input Features")

store = st.sidebar.number_input("Store", 1, 10, 1)
day = st.sidebar.number_input("Day", 1, 31, 1)
month = st.sidebar.number_input("Month", 1, 12, 1)
temp = st.sidebar.slider("Temperature", 0, 50, 30)
fuel = st.sidebar.slider("Fuel Price", 1.0, 5.0, 3.5)
cpi = st.sidebar.slider("CPI", 150, 300, 200)
un = st.sidebar.slider("Unemployment", 1.0, 10.0, 7.0)
hol = st.sidebar.selectbox("Holiday", [0, 1])

tf = temp * fuel

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input Summary")
    st.write({
        "Store": store,
        "Day": day,
        "Month": month,
        "Temp": temp,
        "Fuel": fuel,
        "CPI": cpi,
        "Unemployment": un,
        "Holiday": hol
    })

with col2:

    if st.button("Predict Sales"):

        x = np.array([
            [
                store,
                day,
                month,
                temp,
                fuel,
                cpi,
                un,
                hol,
                tf
            ]
        ])

        p = model.predict(x)

        st.success("Prediction")

        st.metric(
            label="Weekly Sales",
            value=round(p[0], 2)
        )

st.markdown("---")

st.caption("Store Sales AI Project | Final Year ML Project")