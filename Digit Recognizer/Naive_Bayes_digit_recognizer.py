"""
Created on Mon Feb  4 10:15:06 2019
The naive Bayes algorithm is used to predict numerical values within the Digit Recognizer dataset from Kaggle.
@author: mmcclure
"""
# code structure and inspriration taken from
# https://www.kaggle.com/nageshnaik/iris-dataset-classfication-using-naive-bayes

import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

digit_train = pd.read_csv('Kaggle-digit-train.csv')
# Split-out validation dataset
array = digit_train.values
X = digit_train.drop(['label'],axis=1)
Y = digit_train['label']
# One-third of data as a part of test set
validation_size = 0.33
seed = 10
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

# digit_x_train = digit_train.drop(['label'],axis=1)
# digit_y_train = digit_train['label']


#Initialize Gaussian Naive Bayes
clf = GaussianNB()
#Fitting the training set
clf.fit(X_train, Y_train) 
#Predicting the model test set
pred_clf = clf.predict(X_validation)
pred_clf = pd.DataFrame(pred_clf)
# accuracy score
accuracy_train_nb = accuracy_score(Y_train,clf.predict(X_train))
accuracy_validation_nb = accuracy_score(Y_validation,pred_clf)
accuracy_train_nb # 88% accuracy predicting the variables
accuracy_validation_nb # 89% accuracy predicting the outcome ("label")
print(confusion_matrix(Y_validation,pred_clf)) 
print(classification_report(Y_validation,pred_clf)) # 72% accuracy

# Now we can use this on the real test dataset.
digit_test = pd.read_csv('Kaggle-digit-test.csv')
digit_y_test = digit_test.drop(['label'],axis=1)
pred_test = clf.predict(digit_y_test)
pred_test = pd.DataFrame(pred_test)



# Other means of predicting the accuracy
# Prediction Probability
prob_pos_clf = clf.predict_proba(X_validation)[:, 1]
prob_pos_clf = pd.DataFrame(prob_pos_clf)
#Create the prediction file by concatenation of the original data and predictions

#Reshaping needed to perform the concatenation
#pred_clf_df = pd.DataFrame(pred_clf.reshape(50,1))

#Column renaming to indicate the predictions
prob_pos_clf.rename(columns={0:'Prediction'}, inplace=True)

#reshaping the test dataset
#X_validation_df = pd.DataFrame(X_validation.reshape(50,4))

#concatenating the two pandas dataframes over the columns to create a prediction dataset
pred_outcome = pd.concat([X_validation, pred_clf], axis=1, join_axes=[X_validation.index])
#pred_outcome.rename(columns = {0:'SepalLengthCm', 1:'SepalWidthCm', 2:'PetalLengthCm', 3:'PetalWidthCm'}, inplace=True)

#Model Performance
#setting performance parameters
kfold = model_selection.KFold(n_splits=10, random_state=seed)
# Test options and evaluation metric
scoring = 'accuracy'
#calling the cross validation function
cv_results = model_selection.cross_val_score(GaussianNB(), X_train, Y_train, cv=kfold, scoring=scoring)
#displaying the mean and standard deviation of the prediction
msg = "%s: %f (%f)" % ('NB accuracy', cv_results.mean(), cv_results.std())
print(msg) # The model is 57% accurate
# Note that, since this dataset is much larger than the training data, the nature of NB may lead to higher predictions.
# https://jakevdp.github.io/PythonDataScienceHandbook/05.05-naive-bayes.html
