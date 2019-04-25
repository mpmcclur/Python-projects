"""
Created on Thu Jan 31 20:22:44 2019

@author: Matt
"""

import matplotlib.pyplot as plt
import seaborn as sns; sns.set()  # for plot styling
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist
import plotly
import plotly.graph_objs as go

# import data
fed_paper = pd.read_csv("fedPapers85.csv")
# remove 2nd column
del fed_paper["filename"]
# duplicate data to replace Names column with numbers
mapping = {'set': 1, 'test': 2}
fed_paper2 = fed_paper.replace({'Hamilton': 1, 'Madison': 2, 'Jay': 3, 'HM':4, 'dispt':5})
# create variables containing the predictive variables x and the variable to be predicted y
# test and training data are not needed for the K-means clustering approach, but I'm creating them for future use with this dataset
fed_x = fed_paper2.drop(['author'],axis=1) # this variable will be used
fed_x_train = fed_x[12:]
fed_x_test = fed_x[0:11]
fed_y = fed_paper2['author'] # this variable will be used
fed_y_train = fed_y[12:]
fed_y_test = fed_y[0:11]
# test and train data
fed_paper_test = fed_paper2[0:11]
fed_paper_train = fed_paper2[12:]

# make author column index
fed_index = fed_paper2
fed_index.set_index("author", inplace = True)

# just values
fed_values = fed_paper2.values

# sometimes it's important to use dummy variables, e.g.,
# df = pd.get_dummies(df, columns=['column'])
# or to standardize the dat, e.g., 
# df = stats.zscore(df_tr['columns'])

# find out statistics of test and train data
print("***** Train_Set *****")
print(fed_paper_train.describe())
print("***** Test_Set *****")
print(fed_paper_test.describe())

# are there NAs?
fed_paper_train.isna().head() # nope
# if there were, I'd replace them with the mean value in the columns:
# train.fillna(fed_paper_train2.mean(), inplace=True)

# We will start with simple k-means to get a feel for the data and should need only 4 clusters.
# Since Madison, Hamilton, Jay, or Madison & Hamilton are the authors, we want "disputed"" to be clustered into one of the four categories.
# But first, let's run the elbow test to check the potential number of clusters and then proceed with k-means clustering.
distortions = []
K = range(1,7) #arbitarily chosen; just want to get a good idea of how many potential centroids are generated.
for k in K:
    kmeanModel = KMeans(n_clusters=k).fit(fed_x)
    kmeanModel.fit(fed_x)
    distortions.append(sum(np.min(cdist(fed_x, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / fed_x.shape[0])
# Plot the elbow
plt.plot(K, distortions, 'bx-')
plt.xlabel('k')
plt.ylabel('Distortion')
plt.title('The Elbow Method')
plt.show()
# the elbow plot indicates 3, maybe 4, clusters
kmeans = KMeans(n_clusters=4,init='k-means++',max_iter=500,n_init=25,random_state=0) 
y_kmeans = kmeans.fit_predict(fed_x)
fed_x = fed_x.values

 # 1st row as the column names
# visualize clusters
plt.scatter(fed_x[y_kmeans == 0, 0], fed_x[y_kmeans == 0,1],s=100,c='red',label='Cluster 1')
plt.scatter(fed_x[y_kmeans == 1, 0], fed_x[y_kmeans == 1,1],s=100,c='black',label='Cluster 2')
plt.scatter(fed_x[y_kmeans == 2, 0], fed_x[y_kmeans == 2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(fed_x[y_kmeans == 3, 0], fed_x[y_kmeans == 3,1],s=100,c='green',label='Cluster 4')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=500,c='yellow',label='Centroids')
plt.title('Clusters of Authors')
plt.legend()
plt.show()
kmeans.labels_

# Distinct clusters cannot be viewed visually, but what can be done is to view which authors predominantly correspond to which cluster.
y_kmeansdf = pd.DataFrame(y_kmeans)
ykmeans_author = pd.concat([fed_y, y_kmeansdf], axis=1, join_axes=[fed_paper2.index])
# ykmeans_author = ykmeans_author.values
Hamilton_cluster = ykmeans_author.loc[ykmeans_author["author"]==1].agg([np.mode])
Madison_cluster = ykmeans_author.loc[ykmeans_author["author"]==2]
Jay_cluster = ykmeans_author.loc[ykmeans_author["author"]==3]
HM_cluster = ykmeans_author.loc[ykmeans_author["author"]==4]
disp_cluster = ykmeans_author.loc[ykmeans_author["author"]==4]
# find which author appears most often in each cluster
Hamilton_cluster.mode() # Hamilton is cluster 3
Madison_cluster.mode() # Madison is cluster 0
Jay_cluster.mode() # Jay is cluster 2
HM_cluster.mode() # HM is cluster 0
disp_cluster.mode() # disp is cluster 0

# Note that Madison is grouped with Hamilton and Madison.
# Thus, our clustering result suggests the disputed papers were written either by Madison or Hamilton and Madison















# old code
# Visualize the cluster centers
fed_values = fed_paper2.values
kmeans = KMeans(n_clusters=3).fit(fed_values)
plt.scatter(fed_values[:,0], fed_values[:,1], c=kmeans.labels_, cmap='rainbow')
plt.scatter(kmeans.cluster_centers_[:,0] ,kmeans.cluster_centers_[:,1], color='black') 
plt.show()
# another method of viewing center clusters
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=0.5);

# Visualize results
# make_blobs is a great way to make own scatterplot for predefined clusters:
#fed_paper2_plot, y_true = make_blobs(n_samples=300, centers=4,cluster_std=0.60, random_state=0)

# Begin k-means modeling
# K Means Cluster
labels = kmeans.predict(fed_values)
centroids = kmeans.cluster_centers_

colors = ['r','g','b','y','c','m','o','w']
fig2=plt.figure()
kx=fig2.add_subplot(111)

for i in range(k):
        points = np.array([fed_values[j] for j in range(len(fed_values)) if labels[j] ==i])
        kx.scatter(points[:,0],points[:,1],s=20,cmap='rainbow')
kx.scatter(centroids[:,0],centroids[:,1],marker='*',s=200,c='#050505')
print("Final Centroids")
print(centroids)
plt.xlabel('Distance Feature')
plt.ylabel('Speeding Feature')
plt.title('Number cluster={}'.format(k))


# Other k-means example

# example 3D plot
centroids = getTrace(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], kmeans.cluster_centers_[:, 2], s= 8, c = 'yellow', label='Centroids')
t1 = getTrace(fed_x[y_kmeans == 0, 0], fed_x[y_kmeans == 0, 1], fed_x[y_kmeans == 0, 2], s= 4, c='red', label = '1') #match with red=1 initial class
t2 = getTrace(fed_x[y_kmeans == 1, 0], fed_x[y_kmeans == 1, 1], fed_x[y_kmeans == 1, 2], s= 4, c='black', label = '2') #match with black=3 initial class
t3 = getTrace(fed_x[y_kmeans == 2, 0], fed_x[y_kmeans == 2, 1], fed_x[y_kmeans == 2, 2], s= 4, c='blue', label = '3') #match with blue=2 initial class
t4 = getTrace(fed_x[y_kmeans == 3, 0], fed_x[y_kmeans == 3, 1], fed_x[y_kmeans == 3, 2], s= 4, c='green', label = '4') #match with green=0 initial class
t5 = getTrace(fed_x[y_kmeans == 4, 0], fed_x[y_kmeans == 4, 1], fed_x[y_kmeans == 4, 2], s= 4, c='cyan', label = '5') #match with black=3 initial class
x=fed_x[:,0]
y=fed_x[:,1]
z=fed_x[:,2]
showGraph("Authors", "1", [min(x),max(x)], "2", [min(y),max(y)], "3", [min(z)-1,max(z)], [t1,t2,t3,t4,t5,centroids])

# now predict
y_pred = np.array(y_kmeans)
y_pred[y_kmeans == 0] = 1
y_pred[y_kmeans == 1] = 3
y_pred[y_kmeans == 2] = 2
y_pred[y_kmeans == 3] = 0
y_pred[y_kmeans == 4] = 3
plt.scatter(fed_x[y_pred == 0, 0], fed_x[y_pred == 0,1],s=100,c='red',label='Hamilton')
plt.scatter(fed_x[y_pred == 1, 0], fed_x[y_pred == 1,1],s=100,c='black',label='Madison')
plt.scatter(fed_x[y_pred == 2, 0], fed_x[y_pred == 2,1],s=100,c='blue',label='Jay')
plt.scatter(fed_x[y_pred == 3, 0], fed_x[y_pred == 3,1],s=100,c='green',label='HM')
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title('Clusters of Authors')
plt.legend()
plt.show()

# success ratio
from sklearn.metrics import confusion_matrix
y_pred = np.array(y_kmeans)
cm = confusion_matrix(fed_y, y_pred)
print("success ratio : ",success_ratio(cm=cm), "%") # 27% success

def getTrace(x, y, z, c, label, s=2):
    trace_points = go.Scatter3d(
    x=x, y=y, z=z,
    mode='markers',
    marker=dict(size=s, line=dict(color='rgb(0, 0, 0)', width=0.5), color=c, opacity=1),
    name=label
    )
    return trace_points;

def showGraph(title, x_colname, x_range, y_colname, y_range, z_colname, z_range, traces):
    layout = go.Layout(
    title=title,
    scene = dict(
    xaxis=dict(title=x_colname, range = x_range),
    yaxis=dict(title=y_colname, range = y_range),
    zaxis=dict(title=z_colname, range = z_range)
    )
    )

    fig = go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig)

def success_ratio(cm):
    total_success = 0;
    total = 0
    for i in range(0, len(cm)):
        for j in range(0, len(cm[i])):
            if i == j: total_success = total_success + cm[i, j]
            total = total + cm[i, j]
    return (100*total_success)/total
