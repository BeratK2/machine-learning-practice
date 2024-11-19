from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

# Split dataset in features and labels
X = iris.data # features
y = iris.target # labels

# print(X.shape)
# print(y.shape)

# hours of study vs good/bad grades
# train with 8 students (training)
# predict with remaining 2 (testing)
# allows for determining model accuracy

# Train test split (test on 20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)