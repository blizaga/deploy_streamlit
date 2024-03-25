import pandas as pd
import joblib

# Load the model and vectorizer
clf = joblib.load('./model/model_SVC.pkl')
vectorizer = joblib.load('./vectorizer/vectorizer_SVC.pkl')

# Test the model on data test
test_data = pd.read_json("./input/sample-test.in.json", lines=True)
test_data['combined_text'] = test_data['city'] + ' ' + test_data['section'] + ' ' + test_data['heading']
X_test = test_data['combined_text']

X_test_vec = vectorizer.transform(X_test)
y_pred = clf.predict(X_test_vec)

# Save the predictions to a file
test_data['category'] = y_pred
test_data.to_json("./output/sample-test.out.json", orient='records', lines=True)