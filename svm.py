from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()

# Split dataset in features and labels
X = iris.data # features
y = iris.target # labels

classes = ['Iris Setosa', 'Iris Versicolour', 'Iris Virginica']

# Train test split (test on 20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Create SVM model
model = svm.SVC()
model.fit(X_train, y_train)

#print(model)

predictions = model.predict(X_test)
acc = accuracy_score(y_test, predictions)

print("Predictions: ", predictions)
print("Actual: ", y_test)
print("Accuracy: ", acc)

for i in range(len(predictions)):
    print(classes[predictions[i]])