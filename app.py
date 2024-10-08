# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load pre-defined datasets
true_df = pd.read_csv("True.csv")
fake_df = pd.read_csv("Fake.csv")

# Concatenate datasets and create labels
true_df['label'] = 1
fake_df['label'] = 0
df = pd.concat([true_df, fake_df])

# Train test split
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Train models
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_vectorized, y_train)

dt_model = DecisionTreeClassifier()
dt_model.fit(X_train_vectorized, y_train)

rf_model = RandomForestClassifier()
rf_model.fit(X_train_vectorized, y_train)

# Streamlit app
st.title("Fake News Detection App")

# User input
user_input = st.text_area("Enter the news text:")

# Prediction
if st.button("Predict"):
    user_input_vectorized = vectorizer.transform([user_input])
    
    # Logistic Regression Prediction
    log_reg_prediction = log_reg_model.predict(user_input_vectorized)
    st.write("Logistic Regression Prediction:", "True" if log_reg_prediction == 1 else "False")
    
    # Decision Tree Prediction
    dt_prediction = dt_model.predict(user_input_vectorized)
    st.write("Decision Tree Prediction:", "True" if dt_prediction == 1 else "False")
    
    # Random Forest Prediction
    rf_prediction = rf_model.predict(user_input_vectorized)
    st.write("Random Forest Prediction:", "True" if rf_prediction == 1 else "False")
