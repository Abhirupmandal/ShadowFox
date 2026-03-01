import sys
import os

sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            ".."
        )
    )
)

import streamlit as st
import pandas as pd
from src.save_model import load

model = load()

st.set_page_config(
    page_title="Store Sales AI",
    layout="wide"
)



col1, col2 = st.columns([1, 6])

with col1:
  logo_path = os.path.join(
    os.path.dirname(__file__),
    "logo.png"
)

if os.path.exists(logo_path):
    st.image(logo_path, width=80)
with col2:
    st.markdown(
        "# Store Sales AI Dashboard"
    )

st.markdown("---")



tab1, tab2, tab3 = st.tabs(
    ["Prediction", "Charts", "Model Info"]
)



st.sidebar.header("Input")

quantity = st.sidebar.number_input(
    "Quantity", 1, 20, 2
)

discount = st.sidebar.slider(
    "Discount", 0.0, 1.0, 0.1
)

profit = st.sidebar.number_input(
    "Profit", 0, 500, 50
)

category = st.sidebar.selectbox(
    "Category",
    ["Furniture", "Office Supplies", "Technology"]
)

region = st.sidebar.selectbox(
    "Region",
    ["Central", "East", "South", "West"]
)

segment = st.sidebar.selectbox(
    "Segment",
    ["Consumer", "Corporate", "Home Office"]
)



with tab1:

    st.subheader("Prediction Panel")

    col1, col2 = st.columns([1, 1])

    with col1:

        if st.button("Predict Sales"):

            data = {
                "Sales": [0],
                "Quantity": [quantity],
                "Discount": [discount],
                "Profit": [profit],
                "Category": [category],
                "Region": [region],
                "Segment": [segment],
            }

            df = pd.DataFrame(data)

            df = pd.get_dummies(
                df,
                columns=[
                    "Category",
                    "Region",
                    "Segment"
                ]
            )

            df = df.drop("Sales", axis=1)

            df = df.reindex(
                columns=model.feature_names_in_,
                fill_value=0
            )

            pred = model.predict(df)

            result = round(pred[0], 2)

            st.success("Prediction Done")

    with col2:

        if "result" in locals():

            st.metric(
                "Predicted Sales",
                result
            )

            st.info(
                "Predicted using RandomForest Model"
            )



with tab2:

    st.subheader("Charts")

    col1, col2 = st.columns(2)

    if os.path.exists("logs/sales_graph.png"):
        col1.image("logs/sales_graph.png")

    if os.path.exists("logs/importance.png"):
        col2.image("logs/importance.png")


with tab3:

    st.subheader("Model Info")

    st.write("Model: RandomForest")

    st.write("Dataset: Superstore")

    st.write(
        [
            "Quantity",
            "Discount",
            "Profit",
            "Category",
            "Region",
            "Segment",
        ]
    )