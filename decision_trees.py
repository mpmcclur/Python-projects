"""
Created on Mon Feb  4 09:06:15 2019

@author: mmcclure
"""

# import packages
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.externals.six import StringIO  
from IPython.display import Image  
from sklearn.tree import export_graphviz
import pydotplus

### FEDERALIST PAPERS ###
# import data
fed_paper = pd.read_csv("fedPapers85.csv")
# remove 2nd column
del fed_paper["filename"]
# duplicate data to replace Names column with numbers
fed_paper2 = fed_paper.replace({'Hamilton': 1, 'Madison': 2, 'Jay': 3, 'HM':4, 'dispt':5})
# create variables containing the predictive variables x and the variable to be predicted y
fed_x = fed_paper2.drop(['author'],axis=1)
fed_x_train = fed_x[12:]
fed_x_test = fed_x[0:11]
fed_y = fed_paper2['author']
fed_y_train = fed_y[12:]
fed_y_test = fed_y[0:11]
# test and train data
fed_paper_test = fed_paper2[0:11]
fed_paper_train = fed_paper2[12:]

# just values
fed_values = fed_paper2.values

# Note that, unlike in R, there is no complexity parameter (CP) in the python DT library.
# Also note we cannot test the accuracy of this model, since we do not actually know the disputed authors.

# determine the author using a decision tree classifier 
dtc = DecisionTreeClassifier()
dtc_train = dtc.fit(fed_x_train, fed_y_train)
dtc_predict = dtc.predict(fed_x_test)
# the decision tree predicts Madison as the author.

# determine the author using a decision tree regressor
dtr = DecisionTreeRegressor()
dtr_train = dtr.fit(fed_x_train, fed_y_train)
dtr_predict = dtc.predict(fed_x_test)
# also Madison

# Visualize the decision trees
dot_data = StringIO()
export_graphviz(dtc, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
Image(graph.create_png())


# Additional pruning








### DIGIT RECOGNIZER ###
