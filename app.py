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
    top_scores = similarities[top_positions].values
    return list(zip(top_diseases, top_scores))

# -------------------------------
# 3️⃣ Streamlit UI
# -------------------------------
st.title("Medical Disease Prediction App")
st.write("Type and select the symptoms you are experiencing:")

# Searchable multiselect for symptoms
selected_symptoms = st.multiselect("Select Symptoms", options=symptoms)

# Convert selected symptoms to 0/1 vector
user_input_vector = [1 if symptom in selected_symptoms else 0 for symptom in symptoms]

# Predict button
if st.button("Predict Disease"):
    if len(selected_symptoms) == 0:
        st.warning("Please select at least one symptom to predict.")
    else:
        top5 = predict_top_n(user_input_vector, X, diseases, top_n=5)
        st.subheader("Top 5 Predicted Diseases:")
        for disease, score in top5:
            st.write(f"{disease} -> Similarity: {score:.2f}")
