import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
import joblib

# Streamlit UI
st.title('Craigslist Post Category Predictor')

# Input form for user input
uploaded_file = st.file_uploader("Upload JSON File", type="json")
if uploaded_file is not None:
    test_data = pd.read_json(uploaded_file, lines=True)

    st.write("### Sample Data from Uploaded File:")
    st.write(test_data.head())

    # Combine 'city', 'section', and 'heading' into one text feature
    test_data['combined_text'] = test_data['city'] + ' ' + test_data['section'] + ' ' + test_data['heading']

    # Load the vectorizer
    vectorizer = joblib.load('./vectorizer/vectorizer_SVC.pkl')

    # Load the model
    clf = joblib.load('./model/model_SVC.pkl')

    # Predict category for each row in the uploaded file
    predictions = []
    for index, row in test_data.iterrows():
        input_text = row['combined_text']
        input_vec = vectorizer.transform([input_text])
        prediction = clf.predict(input_vec)
        predictions.append(prediction[0])

    # Display predictions in a table
    st.write("### Predictions:")
    result_df = pd.DataFrame({'Category': predictions})
    st.write(result_df)
