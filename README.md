# Student-Habits-Advisor

# Academic Habit Analyzer

A simple and interactive Streamlit web app that uses **K-Means Clustering** to analyze a student's academic and lifestyle habits and recommend personalized suggestions.

---

## Features

- Predicts your academic behavior cluster
- Visual comparison of your habits vs your cluster's average
- Personalized suggestions to improve productivity and well-being
- Mobile-responsive and easy to use

---

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Matplotlib
- Pandas
- NumPy
- Joblib

---

## How It Works

1. Enter your routine details (study hours, sleep, exercise, etc.)
2. App uses a pre-trained **KMeans clustering model** to classify your habit pattern
3. See your cluster label (e.g., *Well-Rounded Achiever*) and how you compare to the cluster average
4. Get smart suggestions tailored to your lifestyle

---

## Setup Instructions (For Local Use)

```bash
git clone https://github.com/your-username/Academic-Habit-Analyzer.git
cd Academic-Habit-Analyzer
pip install -r requirements.txt
streamlit run app.py
