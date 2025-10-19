import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import jaccard_score

# -------------------------------
# 1️⃣ Load dataset
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("training_dataset.csv")  # Replace with your file path
    X = df.drop("disease", axis=1)
    diseases = df["disease"].values
    symptoms = X.columns.tolist()
    return df, X, diseases, symptoms

df, X, diseases, symptoms = load_data()

# -------------------------------
# 2️⃣ Function to predict top N diseases
# -------------------------------
def predict_top_n(user_symptoms, reference_X, reference_diseases, top_n=5):
    """
    Returns top N most similar diseases with similarity scores.
    user_symptoms: 0/1 vector for all symptoms
    """
    user_symptoms = np.array(user_symptoms)
    similarities = reference_X.apply(lambda row: jaccard_score(row.values, user_symptoms), axis=1)
    top_positions = similarities.nlargest(top_n).index
    top_diseases = reference_diseases[top_positions.to_numpy()]
    top_scores = sim_
