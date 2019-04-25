"""
Created on Sat Feb  2 20:34:51 2019 
Multiple machine learning models using four different algorithms, K nearest neighbor, support vector machine (SVM), kernel SVM, and random forest (RF)
@author: Matt
"""
import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor

# load in the data using pandas
digit_train = pd.read_csv('Kaggle-digit-train.csv')
# Dimension of data
digit_train.shape
digit_train.describe()
# Split-out validation dataset
array = digit_train.values
X = digit_train.drop(['label'],axis=1)
Y = digit_train['label']
# One-third of data as a part of test set
validation_size = 0.33
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# Feature Scaling
# this dataset does not require feature scaling / normalziation, but here is the code to do so.
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()  
# X_train = sc.fit_transform(X_train)  
# X_validation = sc.transform(X_validation)  


# KNN Model using k=14
# https://scikit-learn.org/stable/modules/neighbors.html
seed = 10
knn = KNeighborsClassifier(n_neighbors=14)
# fit the model
knn.fit(X_train,Y_train)
# predict
y_pred_knn = knn.predict(X_validation)
# accuracy score
accuracy_train_knn = accuracy_score(Y_train,knn.predict(X_train))
accuracy_validation_knn = accuracy_score(Y_validation,y_pred_knn)
accuracy_train_knn # 88% accuracy predicting the variables
accuracy_validation_knn # 89% accuracy predicting the outcome ("label")
print(confusion_matrix(Y_validation,y_pred_knn)) 
print(classification_report(Y_validation,y_pred_knn)) # 90% accuracy

# Now we can use this on the real test dataset.
digit_test = pd.read_csv('Kaggle-digit-test.csv')
digit_y_test = digit_test.drop(['label'],axis=1)
pred_test_knn = knn.predict(digit_y_test)
pred_test_knn = pd.DataFrame(pred_test_knn)


# SVM Model
# https://scikit-learn.org/stable/modules/svm.html
svclassifier = SVC(kernel='linear')
# fit the model
svclassifier.fit(X_train, Y_train)
# predict
y_pred_svm = svclassifier.predict(X_validation)
# accuracy score
accuracy_train_svm = accuracy_score(Y_train,svclassifier.predict(X_train))
accuracy_validation_svm = accuracy_score(Y_validation,y_pred_svm)
print(confusion_matrix(Y_validation,y_pred_svm))
print(classification_report(Y_validation,y_pred_svm)) # 89% accuracy

# Now we can use this on the real test dataset.
pred_test_svm = svclassifier.predict(digit_y_test)
pred_test_svm = pd.DataFrame(pred_test_svm)


# Kernel SVM Model
# Normal SVM is used to find linearly separable data; kernel SVM considers higher dimensions for this separability; it's more complex.
# first, we'll try the polynomial kernel
k_svclassifier_poly = SVC(kernel='poly', degree=8)  
k_svclassifier_poly.fit(X_train, Y_train)  
# predict
y_pred_k_svm = k_svclassifier_poly.predict(X_validation)
# accuracy score
accuracy_train_k_svm = accuracy_score(Y_train,k_svclassifier_poly.predict(X_train))
accuracy_validation_svm = accuracy_score(Y_validation,y_pred_k_svm)
print(confusion_matrix(Y_validation,y_pred_k_svm))
print(classification_report(Y_validation,y_pred_k_svm)) # 77% accuracy

# second, we'll try the Gaussian kernel
k_svclassifier_gauss = SVC(kernel='rbf')
k_svclassifier_gauss.fit(X_train, Y_train)  
# predict
y_pred_k_svm = k_svclassifier_gauss.predict(X_validation)
# accuracy score
accuracy_train_k_svm = accuracy_score(Y_train,k_svclassifier_gauss.predict(X_train))
accuracy_validation_svm = accuracy_score(Y_validation,y_pred_k_svm)
print(confusion_matrix(Y_validation,y_pred_k_svm))
print(classification_report(Y_validation,y_pred_k_svm)) # 1% accuracy!!

# third, we'll try the sigmoid kernel
k_svclassifier_sig = SVC(kernel='sigmoid') 
k_svclassifier_sig.fit(X_train, Y_train)  
# predict
y_pred_k_svm = k_svclassifier_sig.predict(X_validation)
# accuracy score
accuracy_train_k_svm = accuracy_score(Y_train,k_svclassifier_sig.predict(X_train))
accuracy_validation_svm = accuracy_score(Y_validation,y_pred_k_svm)
print(confusion_matrix(Y_validation,y_pred_k_svm))
print(classification_report(Y_validation,y_pred_k_svm)) # 1% accuracy!!


# Random Forest Model
# 200 trees
# https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
rf = RandomForestClassifier(n_estimators=200, random_state=0) # random_state is random generator number, if blank, then uses np.random generator
rf.fit(X_train, Y_train)
# predict
y_pred_rf = rf.predict(X_validation)
# accuracy score
accuracy_train_rf = accuracy_score(Y_train,rf.predict(X_train))
accuracy_validation_rf = accuracy_score(Y_validation,y_pred_rf)
print(confusion_matrix(Y_validation,y_pred_rf))
print(classification_report(Y_validation,y_pred_rf)) # 91% accuracy

# Now we can use this on the real test dataset.
pred_test_rf = RandomForestClassifier.predict(digit_y_test)
pred_test_rf = pd.DataFrame(pred_test_rf)

# side note, we can use random forest for regression as well
rf_regression = RandomForestRegressor(n_estimators=20, random_state=0)
rf_regression.fit(X_train, Y_train)
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_validation, y_pred_rf))
print('Mean Squared Error:', metrics.mean_squared_error(Y_validation, y_pred_rf))
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(Y_validation, y_pred_rf)))

# Of all the algorithms we used, random forest provided the most accurate prediction
