import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Load model
model = joblib.load('model.pkl')

# Load iris dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
class_names = iris.target_names

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# --- Page config ---
st.set_page_config(page_title="🌼 Iris Classifier", page_icon="🌼")

# --- Sidebar ---
with st.sidebar:
    st.header("📖 About")
    st.image(
        "https://upload.wikimedia.org/wikipedia/commons/5/56/Iris_versicolor_3.jpg",
        caption="Iris Flower",
        use_container_width=True
    )
    st.write("Predict Iris species using ML with interactive visualizations!")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["🏠 Predict", "📊 Insights", "ℹ️ About"])

with tab1:
    st.header("🌸 Input Flower Measurements")

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            sepal_length = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8)
            sepal_width = st.slider("Sepal Width (cm)", 2.0, 4.5, 3.0)
        with col2:
            petal_length = st.slider("Petal Length (cm)", 1.0, 7.0, 4.3)
            petal_width = st.slider("Petal Width (cm)", 0.1, 2.5, 1.3)

        submit = st.form_submit_button("🌟 Predict")

    if submit:
        input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        prediction = model.predict(input_data)[0]
        probs = model.predict_proba(input_data)[0]
        predicted_name = class_names[prediction]

        st.success(f"✅ Predicted Species: **{predicted_name.capitalize()}**")

        # 📊 Probabilities: Bar chart
        prob_df = pd.DataFrame({
            'Species': class_names,
            'Probability': probs
        })
        st.subheader("📊 Prediction Probabilities")
        fig_bar = px.bar(prob_df, x="Species", y="Probability", color="Species", range_y=[0, 1])
        st.plotly_chart(fig_bar)

        # 📈 Probabilities: Pie chart
        st.subheader("🥧 Probability Pie Chart")
        fig_pie = px.pie(prob_df, names='Species', values='Probability',
                         title='Probability Distribution')
        st.plotly_chart(fig_pie)

        # 📍 Scatter plot: User input vs dataset
        st.subheader("📍 Your Flower vs Dataset (Petal Length vs Petal Width)")
        df = pd.DataFrame(X, columns=feature_names)
        df['species'] = [class_names[i] for i in y]

        fig_scatter = px.scatter(
            df, x='petal length (cm)', y='petal width (cm)',
            color='species', title='Iris Dataset',
            opacity=0.7
        )
        fig_scatter.add_scatter(
            x=[petal_length], y=[petal_width],
            mode='markers',
            marker=dict(size=15, color='black'),
            name='Your Flower'
        )
        st.plotly_chart(fig_scatter)

        # 📌 Show input data
        st.write(f"**Your Inputs:**")
        st.write(f"- Sepal Length: {sepal_length} cm")
        st.write(f"- Sepal Width: {sepal_width} cm")
        st.write(f"- Petal Length: {petal_length} cm")
        st.write(f"- Petal Width: {petal_width} cm")

with tab2:
    st.header("📊 Model Insights")

    # Confusion matrix
    st.subheader("📏 Confusion Matrix")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2])

    fig_cm, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues')
    st.pyplot(fig_cm)

    # Feature importance
    st.subheader("🔑 Feature Importance")
    importances = model.feature_importances_
    imp_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
    fig_imp = px.bar(imp_df, x="Feature", y="Importance", color="Importance")
    st.plotly_chart(fig_imp)

    # Pairplot (2D)
    st.subheader("📌 Dataset Pairwise Plot (Scatter Matrix)")
    df = pd.DataFrame(X, columns=feature_names)
    df['species'] = [class_names[i] for i in y]
    fig_pair = px.scatter_matrix(df, dimensions=feature_names, color='species')
    st.plotly_chart(fig_pair)

    acc = model.score(X_test, y_test)
    st.write(f"✅ **Test Accuracy:** {acc*100:.2f}%")

with tab3:
    st.header("ℹ️ How it works")
    st.write("""
    - Input your flower measurements.
    - See the prediction and probabilities.
    - Visualize where your flower fits in the dataset.
    - Check model accuracy and feature importance.
    - Use these insights to understand ML predictions.
    """)
    st.caption("Built with ❤️ using Streamlit, Plotly, and scikit-learn.")
