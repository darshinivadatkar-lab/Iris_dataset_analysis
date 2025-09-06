import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import tree
import os
from PIL import Image

# ---------------- Load Model ----------------
model = joblib.load("iris_model.pkl")

# ---------------- Load Dataset ----------------
df = pd.read_csv("https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv")

# ---------------- App Layout ----------------
st.title("🌸 Iris Flower Prediction & Data Dashboard")

# ---------------- Sidebar ----------------
st.sidebar.header("🌼 Input Flower Measurements")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Optional: feature descriptions
with st.sidebar.expander("ℹ️ Feature Descriptions"):
    st.write("""
    - **Sepal Length**: Length of the outer petal (sepal) in cm  
    - **Sepal Width**: Width of the outer petal (sepal) in cm  
    - **Petal Length**: Length of the inner petal (petal) in cm  
    - **Petal Width**: Width of the inner petal (petal) in cm  
    """)

# ---------------- Prediction ----------------
st.subheader("🌟 Prediction Result")
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

if st.sidebar.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    prediction_clean = prediction.replace("Iris-", "").lower()

    st.success(f"The predicted species is **{prediction_clean.capitalize()}**")
    st.write(descriptions[prediction_clean])

    img_path = images.get(prediction_clean)
    if os.path.exists(img_path):
        st.image(Image.open(img_path), caption=prediction_clean.capitalize(), use_container_width=True)
    else:
        st.warning("Image not found. Please check your `images/` folder.")

# ---------------- Dataset Preview ----------------
st.subheader("📊 Iris Dataset Preview")
rows = st.slider("Select number of rows to view:", 5, len(df), 10)
st.dataframe(df.head(rows))

# ---------------- Data Analysis ----------------
st.subheader("📈 Feature Histograms")
fig, ax = plt.subplots(figsize=(10, 4))
df.drop("species", axis=1).hist(ax=ax, bins=15)
st.pyplot(fig)

st.subheader("🔎 Pairplot by Species")
pairplot_fig = sns.pairplot(df, hue="species")
st.pyplot(pairplot_fig)

#st.subheader("🌳 Decision Tree Visualization")
#clf = model
#fig2, ax2 = plt.subplots(figsize=(12, 8))
#tree.plot_tree(clf, feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"], 
 #              class_names=df["species"].unique(), filled=True)
#st.pyplot(fig2)


from sklearn.tree import plot_tree

st.subheader("🌳 Single Tree from Random Forest")
single_tree = model.estimators_[0]  # first tree in the forest

fig2, ax2 = plt.subplots(figsize=(12, 8))
plot_tree(single_tree, feature_names=["sepal_length", "sepal_width", "petal_length", "petal_width"],
          class_names=df["species"].unique(), filled=True)
st.pyplot(fig2)
