import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
import joblib

# Load the training data
train_data = pd.read_json("./training.json", lines=True)

# Data Preprocessing
# Combine 'city', 'section', and 'heading' into one text feature
train_data['combined_text'] = train_data['city'] + ' ' + train_data['section'] + ' ' + train_data['heading']
X = train_data['combined_text']
y = train_data['category']

# Feature Extraction using TF-IDF
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Model Training
# Using Support Vector Classifier as an example, you can choose any other classifier as per your preference
X_train, X_test, y_train, y_test = train_test_split(X_vec, y, test_size=0.2, random_state=42)
clf = SVC(kernel='linear')
clf.fit(X_train, y_train)

# Model Evaluation
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(clf, 'model.pkl')

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')

