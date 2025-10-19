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
    X = df.drop("Disease", axis=1)
    diseases = df["Disease"].values
    symptoms = X.columns.tolist()
    return df, X, diseases, symptoms

df, X, diseases, symptoms = load_data()

# -------------------------------
# 2️⃣ Function to predict top N diseases
# -------------------------------
def predict_top_n(user_symptoms, reference_X, reference_diseases, top_n=5):
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
st.write("Select the symptoms you are experiencing:")

# Display symptoms as checkboxes in columns
user_input_vector = []
cols = st.columns(4)
for i, symptom in enumerate(symptoms):
    col = cols[i % 4]
    checked = col.checkbox(symptom)
    user_input_vector.append(1 if checked else 0)

# Predict button
if st.button("Predict Disease"):
    top5 = predict_top_n(user_input_vector, X, diseases, top_n=5)
    st.subheader("Top 5 Predicted Diseases:")
    for disease, score in top5:
        st.write(f"{disease} -> Similarity: {score:.2f}")
