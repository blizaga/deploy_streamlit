import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib

# Load the test data
test_data = pd.read_json("./dataset/sample-test.in.json", lines=True)

# Combine 'city', 'section', and 'heading' into one text feature
test_data['combined_text'] = test_data['city'] + ' ' + test_data['section'] + ' ' + test_data['heading']

#load the vectorizer
vectorizer = joblib.load('./vectorizer/vectorizer_SVC.pkl')

# Load the model
clf = joblib.load('./model/model_SVC.pkl')

# Streamlit UI
st.title('Craigslist Post Category Predictor')

# Input form for user input
city = st.selectbox('Select City', test_data['city'].unique())
section = st.selectbox('Select Section', test_data['section'].unique())
heading = st.text_area('Enter Post Heading')

# Predicting category based on user input
if st.button('Predict Category'):
    # Preprocess input text
    input_text = city + ' ' + section + ' ' + heading
    input_vec = vectorizer.transform([input_text])
    
    # Predict category
    prediction = clf.predict(input_vec)
    
    # Display result
    st.write('Predicted Category:', prediction[0])
