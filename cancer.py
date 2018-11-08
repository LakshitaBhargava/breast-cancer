import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import cross_validation, neighbors
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix

df = pd.read_csv('breast-cancer.csv')
df.replace('?', -99999, inplace=True) #missing values
df.drop(['id'], 1, inplace=True)
# Model Features
X = np.array(df.drop(['class'], 1))
# Model labels
y = np.array(df['class'])
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)# train=80%,test=20%
n=10
cls = neighbors.KNeighborsClassifier(n)

cls.fit(X_train, y_train)

accuracy = cls.score(X_test, y_test)

print(accuracy)

y_true=[2,4]
example_measures = np.array([[4, 2, 1, 1, 1, 3, 3, 2, 1], [4, 10, 4, 2, 8, 2, 10, 10, 0]]) # Sample data taken to check
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = cls.predict(example_measures) # predicting in which class data belongs in
c1=confusion_matrix(y_true,prediction) #Making confusion matrix of the data taken
print(prediction)
print(c1)
print("For Test set")
pre=cls.predict(X_test) #predicting value of test set
c=confusion_matrix(y_test,pre)#making confusion matrix of test set
print(c)
print(pre)
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
plt.figure()
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)
plt.xlim(0,11)
plt.ylim(0,11)
plt.title("Classification (k = 3)")
plt.show()
