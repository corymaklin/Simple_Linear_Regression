# import libraries
import numpy
import pandas
import matplotlib.pyplot
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# import dataset
dataset = pandas.read_csv('Salary_Data.csv')
X = dataset.iloc[:, 0].values
X = X.reshape(-1,1)
y = dataset.iloc[:, 1].values
y = y.reshape(-1,1)

# split the dataset into training set and test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.33333, random_state = 0)

# fitting simple linear regression to training set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# visualising the training set results
matplotlib.pyplot.scatter(X_train, y_train, color = 'red')
matplotlib.pyplot.plot(X_train, regressor.predict(X_train), color = 'blue')
matplotlib.pyplot.title('Salary vs Experience (Training set)')
matplotlib.pyplot.xlabel('Years of Experience')
matplotlib.pyplot.ylabel('Salary')
matplotlib.pyplot.show()

# visualing the test set results
matplotlib.pyplot.scatter(X_test, y_test, color = 'red')
matplotlib.pyplot.plot(X_train, regressor.predict(X_train), color = 'blue')
matplotlib.pyplot.title('Salary vs Experience (Training set)')
matplotlib.pyplot.xlabel('Years of Experience')
matplotlib.pyplot.ylabel('Salary')
matplotlib.pyplot.show()