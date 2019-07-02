import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_digits
dataset = load_digits()

X = dataset.data
y = dataset.target

some_digit = X[1100]
some_digit_name = some_digit.reshape(8,8)

plt.imshow(some_digit_name)
plt.show()

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y)

from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train,y_train)
log_reg.score(X_train,y_train)


from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
knn.score(X_train,y_train)
knn.score(X_test,y_test)


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
dtc.score(X_train,y_train)
dtc.predict(X[[1100]])
dtc.score(X_test,y_test)

y_pred = log_reg.predict(X) 

#make confusion matrix 
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y,y_pred)


from sklearn.metrics import precision_score,recall_score,f1_score
precision_score(y,y_pred)
recall_score(y,y_pred)
f1_score(y,y_pred)

