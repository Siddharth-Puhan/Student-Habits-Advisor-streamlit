import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load saved models/data
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
df = joblib.load("dataframe.pkl")
cluster_means = pd.read_csv("cluster_means.csv", index_col=0)
cluster_means.index = cluster_means.index.astype(int)  

# Cluster labels
cluster_labels = {
    0: "Stressed Minimalists",
    1: "Disengaged Energizers",
    2: "Well-Rounded Achievers"
}

# UI layout
st.set_page_config(page_title="Academic Habit Analyzer", layout="centered")
st.title("Academic Habit Analyzer")
st.markdown("Fill in your daily routine details below:")

# Form inputs
study_hours = st.number_input("📚 Study Hours per Day", 0.0, 12.0, step=0.5)
mental_health = st.slider("🧠 Mental Health Rating (0-10)", 0.0, 10.0, 5.0)
exercise = st.slider("💪 Exercise Frequency (days/week)", 0, 7, 3)
attendance = st.slider("📅 Attendance Percentage", 0, 100, 75)
sleep = st.number_input("🛌 Sleep Hours per Day", 0.0, 12.0, step=0.5)
social_media = st.number_input("📱 Social Media Hours", 0.0, 10.0, step=0.5)
netflix = st.number_input("📺 Netflix/Entertainment Hours", 0.0, 10.0, step=0.5)

if st.button("🔍 Predict My Cluster"):
    input_data = [[
        study_hours,
        mental_health,
        exercise,
        attendance,
        sleep,
        social_media,
        netflix
    ]]
    
    scaled_input = scaler.transform(input_data)
    cluster = model.predict(scaled_input)[0]
    cluster_name = cluster_labels.get(cluster, "Unknown Cluster")
    
    st.subheader(f"You belong to: **{cluster_name}**")

    # Suggestions
    suggestions = []
    if study_hours < cluster_means.loc[cluster, 'study_hours_per_day']:
        suggestions.append("📚 Consider increasing your study time.")
    if mental_health < cluster_means.loc[cluster, 'mental_health_rating']:
        suggestions.append("🧠 Focus on improving your mental well-being.")
    if social_media > cluster_means.loc[cluster, 'social_media_hours']:
        suggestions.append("📵 Try to reduce time spent on social media.")
    if exercise < cluster_means.loc[cluster, 'exercise_frequency']:
        suggestions.append("💪 Increase physical activity for better focus.")
    if attendance < cluster_means.loc[cluster, 'attendance_percentage']:
        suggestions.append("📅 Try to attend more classes regularly.")
    if sleep < cluster_means.loc[cluster, 'sleep_hours']:
        suggestions.append("🛌 Improve your sleep routine for better recovery.")
    if netflix > cluster_means.loc[cluster, 'netflix_hours']:
        suggestions.append("🎯 Limit entertainment time during study days.")

    if suggestions:
        st.markdown("###  Personalized Suggestions")
        for s in suggestions:
            st.write(s)

    # Visual comparison
    st.markdown("### 📊 Your Habits vs Cluster Average")
    features = cluster_means.columns.tolist()
    cluster_avg = cluster_means.loc[cluster].values.tolist()

x = np.arange(len(features))
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(x - 0.2, input_data[0], width=0.4, label="You", color="#4CAF50")
ax.bar(x + 0.2, cluster_avg, width=0.4, label="Cluster Avg", color="#2196F3")
ax.set_xticks(x)
ax.set_xticklabels(features, rotation=45)
ax.legend()

for i, v in enumerate(input_data[0]):
    ax.text(x[i] - 0.2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

for i, v in enumerate(cluster_avg):
    ax.text(x[i] + 0.2, v + 0.02, f"{v:.2f}", ha='center', va='bottom', fontsize=8)

st.pyplot(fig)

