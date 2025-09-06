import streamlit as st
import pandas as pd
import joblib
import os

# Load dataset & model
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")
model = joblib.load("iris_model.pkl")

# App title
st.title("🌸 Iris Flower Prediction Dashboard")

st.markdown(
    """
    This app predicts the type of **Iris flower** based on measurements.  
    Adjust the values below and click **Predict**.
    """
)

# Sidebar for dataset preview
st.sidebar.header("🔍 Explore Dataset")
rows = st.sidebar.slider("Select number of rows to view:", 5, len(df), 10)
st.sidebar.dataframe(df.head(rows))

# User inputs
st.subheader("Enter Flower Measurements")
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", 0.0, 10.0, 5.1)
    petal_length = st.number_input("Petal Length (cm)", 0.0, 10.0, 1.4)

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", 0.0, 10.0, 3.5)
    petal_width = st.number_input("Petal Width (cm)", 0.0, 10.0, 0.2)

# Prediction
if st.button("🌼 Predict"):
    features = [[sepal_length, sepal_width, petal_length, petal_width]]
    prediction = model.predict(features)[0]

    st.subheader("🌟 Prediction Result:")
    st.success(f"The predicted species is **{prediction.capitalize()}**")

    # Species descriptions & images
    descriptions = {
        "setosa": "Iris Setosa is the smallest species, with short petals and sepals.",
        "versicolor": "Iris Versicolor is medium-sized, often violet-blue in color.",
        "virginica": "Iris Virginica is the largest, with long petals and sepals."
    }

    images = {
        "setosa": "images/setosa.jpg",
        "versicolor": "images/versicolor.jpg",
        "virginica": "images/virginica.jpg"
    }

    st.write(descriptions[prediction])

    img_path = images.get(prediction)
    if os.path.exists(img_path):
        st.image(img_path, caption=prediction.capitalize(), use_container_width=True)
    else:
        st.warning("Image not found. Please check your `images/` folder.")

