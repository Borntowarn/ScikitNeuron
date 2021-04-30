# pylint: disable=no-member
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.metrics import accuracy_score 

iris = datasets.load_iris()

x = iris.data[:, 1:3]
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, random_state = 0)

sc = StandardScaler()

sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

perc = Perceptron(max_iter = 40, eta0 = 0.1, random_state=0)
perc.fit(x_train_std, y_train)

result = perc.predict(x_test_std)
print ("Число ошибок:", np.where(y_test != result, 1, 0).sum(), sep=" ")
print ("Точность:", round(accuracy_score(y_test, result) * 100) / 100, "%", sep=" ")