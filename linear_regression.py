from sklearn import datasets
from sklearn import linear_model
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# Load boston housing dataset
boston = datasets.fetch_california_housing()

# Features and labels
X = boston.data
y = boston.target

# Create linear regression model
l_reg = linear_model.LinearRegression()

#plt.scatter(X.T[0], y)
#plt.show()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = l_reg.fit(X_train, y_train)
predictions = model.predict(X_test)

print("predictions: ", predictions)
print("Score: ", l_reg.score(X,y))
print("Coedd: ", l_reg.coef_)
print("Intercept: ", l_reg.intercept_)