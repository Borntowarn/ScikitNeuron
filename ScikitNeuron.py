# pylint: disable=no-member
from sklearn import datasets
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score 

iris = datasets.load_iris()

x1 = []
y1 = []
with open("C:\\Users\\kozlo\\source\\repos\\VSCODE\\Neuron\\NewNeuron\\data.txt", "r") as f:
	for a in f:
		a = a.strip().split()
		x1.append(a[0:4])
		if a[4] == "setosa":
			y1.append(0)
		else:
			y1.append(1)

x1 = np.array(x1)
x1 = x1[:, 0:3:2]
y1 = np.array(y1)
#x = iris.data[0:50, 0:3:2]
#y = iris.target[0:100]

#x = np.append(x, iris.data[100:150, 0:3:2], axis=0)

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size = 0.3, random_state = 0)

sc = StandardScaler()

sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

perc = Perceptron(max_iter = 40, eta0 = 0.1, random_state=0)
perc.fit(x_train_std, y_train)

logreg = LogisticRegression(random_state=0, C=1000.0)
logreg.fit(x_train_std, y_train)

result = logreg.predict(x_test_std)
print ("Число ошибок:", np.where(y_test != result, 1, 0).sum(), sep=" ")
print ("Точность:", round(accuracy_score(y_test, result) * 100) / 100, "%", sep=" ")

a = [float(s) for s in input().split()]

while(a[0] != 0):
	a = np.atleast_2d(a)
	a = sc.transform(a)
	result = logreg.predict_proba(a)
	#if (result[0] >= 0.5) : print("setosa c вероятностью ", round(odd*10000)/100, "%")
	#else : print ("versicolor c вероятностью ", round((1-odd)*10000)/100, "%")
	a = [float(s) for s in input().split()]