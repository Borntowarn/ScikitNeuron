# pylint: disable=no-member
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import KFold
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt 
from tqdm import tqdm
import torch as torch

iris = datasets.load_iris()

x1 = iris.data
y1 = iris.target
lambdas = [1e-10,1e-4,1e-3,1e-2,0.1,1.0,1e+2,1e+3,1e+4,1e+10]
final_scores = []
models = []

x_train, x_test, y_train_first, y_test_first = train_test_split(np.array(x1, dtype=float),np.array(y1, dtype=int), test_size=0.3, random_state=0)

sc = StandardScaler()

sc.fit(x_train)
x_train_std = sc.transform(x_train)
x_test_std = sc.transform(x_test)

loo = KFold(7)
for llambda in lambdas:
	lambda_score = []
	for train, test in tqdm(loo.split(x_train_std), total = loo.get_n_splits(x_train_std)):

		x_train, x_test = x_train_std[train], x_train_std[test]
		y_train, y_test = y_train_first[train], y_train_first[test]

		logreg = LogisticRegression(max_iter = 100, penalty = "l2", random_state=0, C=1/llambda)
		logreg.fit(x_train, y_train)

		lambda_score.append(logreg.score(x_test, y_test))
	final_scores.append(np.mean(lambda_score))
	models.append(logreg)
	
print(final_scores)

print(final_scores.index(max(final_scores)))

figure, axes = plt.subplots(nrows=2, ncols=5)
axes = axes.flatten()
x1 = np.array(x1)
x = sc.transform(x1)
colors = ['red', 'green', 'blue']
x_ax = np.linspace(-3,3,100)

for ax,logreg,_lambda in zip(axes,models, lambdas):
	result = logreg.predict(x_test_std)
	a = np.hstack((logreg.intercept_[:,None], logreg.coef_))
	for i in np.unique(y1):
		if i == 2:
			ax.plot(x[y1==i,0], x[y1==i,2], 'o', label="Число ошибок:" + str(np.where(y_test_first != result, 1, 0).sum()), color = colors[i])
			y_ax = ((-a[i][0]-a[i][1]*x_ax)/a[i][3])
			ax.plot(x_ax,y_ax, color = colors[i], label ="Точность:" + str(round(accuracy_score(y_test_first, result) * 10000) / 100) + "%")
		else:
			ax.plot(x[y1==i,0], x[y1==i,2], 'o', color = colors[i])
			y_ax = ((-a[i][0]-a[i][1]*x_ax)/a[i][3])
			ax.plot(x_ax,y_ax, color = colors[i])
	ax.set_title("lambda = " + str(_lambda))
	ax.legend()

plt.show()

a = [float(s) for s in input().split()]

while(a[0] != 0):
	a = np.atleast_2d(a)
	a = sc.transform(a)
	result = logreg.predict_proba(a)
	a = [float(s) for s in input().split()]