import numpy as np
import pandas as pd
from sklearn import neighbors, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv('car.data')

X = data[[
    'buying',
    'maint',
    'safety'
]].values

y = data[['class']]


# Label encoder to convert feature data to numerical format
Le = LabelEncoder()
for i in range(len(X[0])):
    X[:, i] = Le.fit_transform(X[:, i])
#print(X)


# Mapping to convert label data to numerical format
label_mapping = {
    'unacc': 0,
    'acc': 1,
    'good': 2,
    'vgood': 3
}

y['class'] = y['class'].map(label_mapping)
y = np.array(y) # Convert to numpy array
#print(y)


# Create KNN model (25 closest data points)
knn = neighbors.KNeighborsClassifier(n_neighbors=25, weights='uniform')

# Train the model (Test on 20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
knn.fit(X_train, y_train)
prediction = knn.predict(X_test)
accuracy = metrics.accuracy_score(y_test, prediction)

# print("Predictions: ", prediction)
# print("Accuracy: ", accuracy)

print("actual value: ", y[20])
print("predicted value: ", knn.predict(X)[20])