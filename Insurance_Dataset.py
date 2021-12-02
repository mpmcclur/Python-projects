"""
Created on Wed Apr  7 14:58:08 2021

@author: blubb
"""

import pandas as pd
import seaborn as sn
import matplotlib.pyplot as plt
import numpy as np
from functools import reduce
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

file = 'Example_Application_Dataset.xlsx'
xl = pd.ExcelFile(file)
print(xl.sheet_names)
data = xl.parse('Data')
# Remove Division since all are cargo.
data = data.drop('Division',1)

# Col names
for col in data.columns:
    print(col) 
    
# Create dataset without missing data
clean = data.dropna()
clean = clean[clean['BrokerGroupID'] != 'NONE/NA']
# Convert BrokerGroupID to 'int' data type
clean = clean.astype({'BrokerGroupID':'int'})

# 1. Top 10 Broker Groups with Highest Number of Binding Quotes
broker_group = clean['BrokerGroupID'].value_counts()
broker_group = broker_group.iloc[:10].plot(kind="barh")
broker_group.set_ylabel('Broker Group ID')
broker_group.set_title('Top 10 Broker Groups with Bound Quotes')
broker_group.set_xlabel('Total Number of Binding Quotes')

# 2. Top 10 Broker Cities with Highest Number of Binding Quotes¶
clean['Broker City'].value_counts()
city = clean['Broker City'].value_counts()
city_chart = city.iloc[:10].plot(kind="barh")
city_chart.set_ylabel('Cities')
city_chart.set_title('Top 10 Cities with Binding Quotes')
city_chart.set_xlabel('Number of Binding Quotes')
plt.show()

# 3. Comparison of Applications and Quotes Received per Year
# Create 3 new dfs for enter date, quote date, and bound date.
# enter date df
enter_date = data[['application_EnteredDate', 'applicationid', 'AssuredID']].copy()
# consolidate by year and sum total number of applications for each year
enter_date = enter_date['application_EnteredDate'].groupby([enter_date.application_EnteredDate.dt.year]).agg('count')
# convert series to df
enter_date = enter_date.to_frame()
# rename index and column
enter_date.index.names = ['Year']
enter_date.columns = ['Applications Received']

# quote date df
quote_date = data[['DateQuoteEntered', 'applicationid', 'AssuredID']].copy()
# remove NAs
quote_date = quote_date.dropna()
# consolidate by year and sum total number of quotes provided per year.
quote_date = quote_date['DateQuoteEntered'].groupby([quote_date.DateQuoteEntered.dt.year]).agg('count')
# convert series to df
quote_date = quote_date.to_frame()
# rename index and column
quote_date.index.names = ['Year']
quote_date.columns = ['Quotes Provided']

# bound date df
bound_date = data[['DateQuoteBound', 'applicationid', 'AssuredID']].copy()
# remove NAs
bound_date = bound_date.dropna()
# consolidate by year and sum total number of applications for each year
bound_date = bound_date['DateQuoteBound'].groupby([bound_date.DateQuoteBound.dt.year]).agg('count')
# convert series to df
bound_date = bound_date.to_frame()
# rename index and column
bound_date.index.names = ['Year']
bound_date.columns = ['Binding Quotes']

# Merge all three dataframes and then plot
dates = [enter_date, quote_date, bound_date]
dates = reduce(lambda left,right: pd.merge(left,right,on='Year'), dates)
# append 2021 data for binding quotes
#dates = dates.append(bound_date.iloc[3])
# replace NAs with zero for 2021
#dates = dates.replace('NaN', 0)

# Plot the merged df
dates_plot = dates.plot(kind="bar")
dates_plot.set_ylabel('Count')
dates_plot.set_title('Applications and Quotes Per Year')
for p in dates_plot.patches:
    dates_plot.annotate(str(int(p.get_height())),(p.get_x()+p.get_width()/2.,p.get_height()),ha='center',va='center',xytext=(0, 10), textcoords='offset points')
plt.show()

# 4. Duration of Days between Application Statuses
# calculate difference between two date columns as a new column
clean['Quote-Application-Duration'] = clean.DateQuoteEntered-clean.application_EnteredDate
clean['Binding-Quote-Duration'] = clean.DateQuoteBound-clean.DateQuoteEntered
clean['Binding-Application-Duration'] = clean.DateQuoteBound-clean.application_EnteredDate

# combine into new df
clean_duration = pd.DataFrame({'Quote-Application-Duration': clean['Quote-Application-Duration'], 'Binding-Quote-Duration': clean['Binding-Quote-Duration'], 'Binding-Application-Duration': clean['Binding-Application-Duration']})
#clean['Quote-Application-Duration'] = pd.to_datetime(clean['Quote-Application-Duration']).dt.date

# summary statistics of application and quote durations
clean_stats = clean_duration.describe()
clean_stats

# 5. Logistic Regression to Predict Bound Quote
# remove NAs from BrokerGroupID column
data_trim = data[['applicationid', 'AssuredID', 'BrokerGroupID','DateQuoteBound','Broker City','UnderwriterID']].copy()
data_trim = data_trim[data_trim['BrokerGroupID'] != 'NONE/NA']
# convert DateQuoteBound to brinary. date exists = 1, NaN = 0
data_trim['DateQuoteBound'].loc[~data_trim['DateQuoteBound'].isnull()] = 1  # not nan
data_trim['DateQuoteBound'].loc[data_trim['DateQuoteBound'].isnull()] = 0   # nan
data_trim['DateQuoteBound'] = pd.to_numeric(data_trim['DateQuoteBound']) # convert new binary data to int type
# label encode Broker City
data_trim['Broker City'] = data_trim['Broker City'].astype('category') # convert variable to "category" type
data_trim['Broker City'] = data_trim['Broker City'].cat.codes # encode the variable using cat.codes accessor
data_trim.head() # first 10 rows of converted dataframe
#data_trim.dtypes

# Logistic regression using DateQuoteBound as response (dependent) variable
X = data_trim[['applicationid', 'AssuredID', 'BrokerGroupID','Broker City','UnderwriterID']]
X = X.dropna() # remove NAs
X = X.reset_index() # rest index to match y
del X['index']
y = data_trim['DateQuoteBound'] # reset index to match X
y.drop(y.tail(97).index,inplace = True) # delete rows to compensate for NAs deletion
y = y.reset_index()
del y['index']

import statsmodels.api as sm
logit_model=sm.Logit(y,X.astype(float))
result=logit_model.fit()
print(result.summary2())
np.asarray(y)

X = X[['AssuredID', 'BrokerGroupID']]
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=0)
from sklearn.metrics import confusion_matrix
logistic_regression= LogisticRegression()
logistic_regression.fit(X_train,y_train)
y_pred=logistic_regression.predict(X_test)
confusion_matrix = confusion_matrix(y_test, y_pred)
sn.heatmap(confusion_matrix, annot=True)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))
plt.show()

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()