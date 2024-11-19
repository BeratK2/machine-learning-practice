from sklearn.datasets import load_breast_cancer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
import pandas as pd

# Breast cancer dataset
bc = load_breast_cancer()
x = scale(bc.data) # Return dataset as a list and scale it to make it more readable
y = bc.target

# Train KMeans model
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

model = KMeans(n_clusters=2, random_state=0)
model.fit(x_train)

# Predict with testig features
predictions = model.predict(x_test)
labels = model.labels_

print('Labels: ', labels)
print('Predictions: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
print('Actual: ', y_test)