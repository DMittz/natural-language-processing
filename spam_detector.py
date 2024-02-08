# -*- coding: utf-8 -*-
"""Spam Detector

Original file is located at
    https://colab.research.google.com/drive/18ZoaIotr-9QDZpJn4cJrBJEPhXtwswtJ
"""

# Loading the Libraries:
!pip install -q kaggle
! mkdir ~/.kaggle
! cp kaggle.json ~/.kaggle/
! chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d balaka18/email-spam-classification-dataset-csv

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

"""## Reading the Data:"""

!unzip *.zip && rm -r *.zip

df = pd.read_csv("emails.csv")
df.head(10)

sum(df.isnull().sum())

"""##Building the models:"""

X = df.iloc[:,1:3001]
Y = df.iloc[:,-1].values

train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size = 0.20, random_state = 42)

sss = StratifiedShuffleSplit(n_splits = 1, test_size = 0.25, random_state = 42)
train_index, test_index = next(sss.split(X, Y))
x_train, x_test = X.loc[train_index], X.loc[test_index]
y_train, y_test = Y[train_index], Y[test_index]

"""SSS removes chance of bias, equal distribution of 0

## Testing the models:
"""

model = MultinomialNB(alpha = 1.0)
model.fit(train_x,train_y)
y_pred = model.predict(test_x)
print("Accuracy Score for Naive Bayes: ", (accuracy_score(y_pred,test_y) * 100),"%")

model.fit(x_train,y_train)
predictions = model.predict(x_test)
print("Accuracy Score for Stratified Shuffle Split: ", (accuracy_score(predictions,y_test) * 100),"%")

"""SVC gave 91.4 % accuracy, hence removed.

##User Testing:
"""

def word_count(word: str, sentence: str):
  i = 1
  count = 0
  for x in sentence:
    if x == word:
      count += 1
  return count

string = input("Enter:")
words = df.columns
sentence = string.split()
print(sentence)

df1 = df.drop(columns = ['Email No.', 'Prediction'])

occurence = []
for word in df1:
  counter = word_count(word, sentence)
  occurence.append(counter)
print(occurence)

occ = [words, occurence]
occ1 = pd.DataFrame([occurence], columns=df1.columns)
final = model.predict(occ1)
print(final)

"""0 - Not a spam

1 - Spam
"""
