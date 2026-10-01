import numpy as np
import pandas as pd
import streamlit as st
from sklearn.linear_model import LinearRegression


# -------------------------------------------------
# PAGE SETUP
# -------------------------------------------------
st.set_page_config(
    page_title="Student Score Prediction", page_icon="🎓", layout="centered"
)
st.title("🎓 Student Score Prediction")
st.write(
    "Predict a student's exam score based on study hours (1 to 100 hours)."
)


# -------------------------------------------------
# FULL DATASET (24 HOURS & SCORES)
# -------------------------------------------------
data = {
    "Hours": [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        12,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
    ],
    "Score": [
        35,
        40,
        45,
        50,
        55,
        60,
        62,
        65,
        68,
        70,
        72,
        75,
        78,
        80,
        82,
        85,
        87,
        90,
        92,
        95,
        96,
        97,
        98,
        99,
    ],
}
df = pd.DataFrame(data)


# -------------------------------------------------
# TRAIN MODEL ON DATASET
# -------------------------------------------------
X = df[["Hours"]]
Y = df["Score"]


model = LinearRegression()
model.fit(X, Y)


# -------------------------------------------------
# INPUT SECTION (ALLOWED UP TO 100 HOURS)
# -------------------------------------------------
st.subheader("Enter Student Details")
hours = st.number_input(
    "Hours Studied",
    min_value=1.0,
    max_value=100.0,
    value=24.0,
    step=1.0,
)


# -------------------------------------------------
# PREDICTION (CAPPED AT MAXIMUM 100)
# -------------------------------------------------
if st.button("Predict Score", type="primary", use_container_width=True):
  new_student = pd.DataFrame({"Hours": [hours]})


  # Calculate raw prediction from model
  raw_score = model.predict(new_student)[0]


  # Enforce strict maximum score cap at 100
  predicted_score = min(100.0, raw_score)


  st.success(f"Predicted Exam Score: {predicted_score:.2f}")
  st.balloons()
  st.write(f"**Hours Studied:** {hours:.0f}")


# -------------------------------------------------
# DATASET VIEW
# -------------------------------------------------
st.divider()
with st.expander("View Training Dataset"):
  st.dataframe(df, use_container_width=True)