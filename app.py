
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

st.set_page_config(page_title="ReFill Hub Consumer Insights", layout="wide")

st.title("ReFill Hub Consumer Insights & Adoption Prediction Dashboard")

df = pd.read_csv("ReFillHub_SyntheticSurvey.csv")
df['Adopt_Label'] = df['Try_Refill_Likelihood'].apply(lambda x: 1 if x >= 4 else 0)

encoded = df.copy()
for col in encoded.columns:
    if encoded[col].dtype == 'object':
        le = LabelEncoder()
        encoded[col] = le.fit_transform(encoded[col].astype(str))

st.sidebar.header("Filters")
age_filter = st.sidebar.multiselect("Age Group", df['Age_Group'].unique())
income_filter = st.sidebar.multiselect("Income", df['Income'].unique())
eco_slider = st.sidebar.slider("Eco Score (Reduce Waste)", 1, 5, (1,5))

filtered = df.copy()
if age_filter:
    filtered = filtered[filtered['Age_Group'].isin(age_filter)]
if income_filter:
    filtered = filtered[filtered['Income'].isin(income_filter)]
filtered = filtered[(filtered['Reduce_Waste_Score'] >= eco_slider[0]) & (filtered['Reduce_Waste_Score'] <= eco_slider[1])]

tab1, tab2, tab3 = st.tabs(["📊 Dashboard Insights", "🤖 ML Models", "📥 Upload & Predict"])

with tab1:
    st.subheader("Age vs Spending vs Adoption")
    fig, ax = plt.subplots()
    sns.scatterplot(data=filtered, x='Age_Group', y='Willingness_to_Pay_AED', hue='Adopt_Label', ax=ax)
    st.pyplot(fig)

    st.subheader("Income vs Eco Score vs Adoption")
    fig, ax = plt.subplots()
    sns.heatmap(encoded[['Income','Reduce_Waste_Score','Adopt_Label']].corr(), annot=True, cmap="coolwarm")
    st.pyplot(fig)

    st.subheader("Product Preferences & Adoption")
    fig, ax = plt.subplots()
    sns.countplot(data=filtered, x='Preferred_Packaging', hue='Adopt_Label')
    plt.xticks(rotation=45)
    st.pyplot(fig)

    st.subheader("Refill Frequency vs Spending")
    fig, ax = plt.subplots()
    sns.boxplot(data=filtered, x='Usage_Frequency', y='Willingness_to_Pay_AED')
    st.pyplot(fig)

    st.subheader("Plastic Awareness vs Adoption")
    fig, ax = plt.subplots()
    sns.barplot(data=filtered, x='Likely_to_Use_ReFillHub', y='Adopt_Label')
    st.pyplot(fig)

with tab2:
    st.subheader("Machine Learning Model Comparison")

    df_ml = encoded.copy()
    X = df_ml.drop(['Adopt_Label'], axis=1)
    y = df_ml['Adopt_Label']

    models = {
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "Gradient Boosting": GradientBoostingClassifier()
    }

    results = []

    for name, model in models.items():
        model.fit(X, y)
        y_pred = model.predict(X)
        precision = precision_score(y, y_pred)
        recall = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        acc = accuracy_score(y, y_pred)
        auc = roc_auc_score(y, model.predict_proba(X)[:,1])
        results.append([name, acc, precision, recall, f1, auc])

    st.write(pd.DataFrame(results, columns=["Model", "Accuracy", "Precision", "Recall", "F1 Score", "AUC"]))

with tab3:
    st.subheader("Upload New Data for Prediction")
    uploaded = st.file_uploader("Upload CSV", type=['csv'])

    if uploaded:
        new_df = pd.read_csv(uploaded)
        model = joblib.load("model.pkl")

        for col in new_df.columns:
            if new_df[col].dtype == 'object':
                le = LabelEncoder()
                new_df[col] = le.fit_transform(new_df[col].astype(str))

        preds = model.predict(new_df)
        new_df['AdoptionPrediction'] = preds
        st.write(new_df)
        st.download_button("Download Predictions", data=new_df.to_csv(), file_name="Predictions.csv")
