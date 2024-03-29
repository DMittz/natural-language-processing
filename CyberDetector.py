# -*- coding: utf-8 -*-
"""CyberDetector.ipynb

Original file is located at
    https://colab.research.google.com/drive/1XLTWfpYgRT-jfWtPjlUjm9WoQm1VZzva
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow as tf

data = pd.read_csv("CyberBullyingCommentsDataset.csv")

x = data['Text']
y = data['CB_Label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=5)
x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

param_dist = {
    'max_depth': randint(1, 100),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 20),}

# Decision Tree Classifier
dtc = DecisionTreeClassifier(random_state=42)
random_search = RandomizedSearchCV(dtc, param_distributions=param_dist, n_iter=20, cv=5, scoring='accuracy', random_state=42)
random_search.fit(x_train_vec, y_train)

print("Best Parameters:", random_search.best_params_)
best_dtc = random_search.best_estimator_

y_pred_dt = best_dtc.predict(x_test_vec)

accuracy_dt = accuracy_score(y_test, y_pred_dt)
print("Decision Tree Classifier Accuracy: {:.3f}%".format(accuracy_dt * 100))
print("Classification Report for Decision Tree Classifier:")
print(classification_report(y_test, y_pred_dt))

# Neural Network model
model = Sequential([
    Dense(64, activation='relu', input_shape=(x_train_vec.shape[1],)),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')])

x_train_array = x_train_vec.toarray()
x_test_array = x_test_vec.toarray()

model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

model.fit(x_train_array, y_train, epochs=5, batch_size=40, verbose=1)

loss, accuracy_nn = model.evaluate(x_test_array, y_test)
print("Neural Network Accuracy:", accuracy_nn)

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize the TF-IDF vectorizer
vectorizer = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)

x_train_vec = vectorizer.fit_transform(x_train)
x_test_vec = vectorizer.transform(x_test)

# Predictions
y_pred_nn = (model.predict(x_test_vec) > 0.5).astype("int32")
print("Classification Report for Neural Network model:")
print(classification_report(y_test, y_pred_nn))

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(vocabulary=vectorizer.vocabulary_)
vectorizer.fit(x_train)

# Function to preprocess user input and make predictions
def predict_user_input(user_input):
    # Preprocess
    user_input_preprocessed = vectorizer.transform([user_input])

    output = best_dtc.predict(user_input_preprocessed)
    return output

# Get user input
user_input = input("Enter your message: ")

prediction = predict_user_input(user_input)

if prediction == 0:
    print("The message is not harmful.")
else:
    print("The message is harmful.")
