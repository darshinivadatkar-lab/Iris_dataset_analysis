# app.py
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree

# --------------------------
# Load model
# --------------------------
model = joblib.load("iris_model.pkl")  # your trained model

# Load Iris dataset for visualizations
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)

X = df[iris.feature_names]
y = df['species']

# Train a decision tree for visualization
dt_model = DecisionTreeClassifier()
dt_model.fit(X, y)

# --------------------------
# Streamlit App Layout
# --------------------------
st.set_page_config(page_title="Iris Flower Predictor", layout="wide")
st.title("🌸 Iris Flower Predictor")
st.write("Predict the species of an Iris flower and explore data analysis visualizations.")

# --------------------------
# Sidebar input fields
# --------------------------
st.sidebar.header("Input Flower Measurements")
sepal_length = st.sidebar.number_input("Sepal Length (cm)", min_value=0.0, step=0.1, value=float(df['sepal length (cm)'].mean()))
sepal_width  = st.sidebar.number_input("Sepal Width (cm)", min_value=0.0, step=0.1, value=float(df['sepal width (cm)'].mean()))
petal_length = st.sidebar.number_input("Petal Length (cm)", min_value=0.0, step=0.1, value=float(df['petal length (cm)'].mean()))
petal_width  = st.sidebar.number_input("Petal Width (cm)", min_value=0.0, step=0.1, value=float(df['petal width (cm)'].mean()))

# --------------------------
# Prediction with button
# --------------------------
st.subheader("Prediction")
if st.sidebar.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    st.success(f"🌸 Predicted species: **{prediction}**")

# --------------------------
# Dataset display
# --------------------------
st.subheader("Dataset Preview")
st.dataframe(df)

# --------------------------
# Pairplot
# --------------------------
st.subheader("Pairplot of Iris Dataset")
sns_plot = sns.pairplot(df, hue="species", corner=True)
st.pyplot(sns_plot)

# --------------------------
# Histograms in two columns
# --------------------------
st.subheader("Histograms of Features")
fig1, ax1 = plt.subplots(figsize=(6,4))
sns.histplot(df['sepal length (cm)'], kde=True, ax=ax1)
sns.histplot(df['sepal width (cm)'], kde=True, ax=ax1)

fig2, ax2 = plt.subplots(figsize=(6,4))
sns.histplot(df['petal length (cm)'], kde=True, ax=ax2)
sns.histplot(df['petal width (cm)'], kde=True, ax=ax2)

col1, col2 = st.columns(2)
with col1:
    st.pyplot(fig1)
with col2:
    st.pyplot(fig2)

# --------------------------
# Decision Tree Visualization
# --------------------------
st.subheader("Decision Tree Classifier")
plt.figure(figsize=(12,8))
plot_tree(dt_model, feature_names=iris.feature_names, class_names=iris.target_names, filled=True)
st.pyplot(plt)
