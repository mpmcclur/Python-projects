"""
Matt McClure
Using semi-structured JSON data, which is loaded via MongoDB, I explore the Grades dataset by printing basic statistical information,
visualizing the data, and exporting to CSV specific aggregated data.
"""

import pymongo
import urllib.request
import json
import pandas as pd
import matplotlib.pyplot as pp
import statistics as st
import numpy as np

# # Connect to the database
from pymongo import MongoClient
client = MongoClient('localhost', 27017)

# read data from file and insert into collection:
db = client['students_db']
collection_grades = db['grades']
grades = [] # create empty grades list
# open grades data file and import in MDB database; for some reason there was extra data in the JSON file, which makes importing a little more cumbersome
with open('C:/Users/Matt/Desktop/IST 652 (Scripting for Data Analysis)/MongoDB JSON Data/grades.json') as f:
    for line in f:
        file_data = grades.append(json.loads(line))


# Begin preprocessing the data
# transform data into a dataframe by normalizing the data
from pandas.io.json import json_normalize
#panda_grades = pd.DataFrame.from_dict(grades, orient='columns')
panda_grades = pd.DataFrame.from_dict(json_normalize(grades), orient='columns')
# next, "decode" the scores variables
scores = panda_grades.scores
# transform to dictionary
scores = scores.to_dict()
# break up and transform into dataframe
scores = panda_grades['scores'].apply(pd.Series)
scores2 = scores[0].apply(pd.Series)
scores3 = scores[1].apply(pd.Series)
scores4 = scores[2].apply(pd.Series)
scores5 = scores[3].apply(pd.Series)
scores6 = scores[4].apply(pd.Series)
scores7 = scores[5].apply(pd.Series)
#scores2 = pd.DataFrame.from_dict(json_normalize(scores2), orient='columns')
# combine the scores
scores_final = pd.concat([scores2,scores3,scores4,scores5,scores6,scores7], axis=1, join_axes=[scores2.index])
# delete dud columns
scores_final.drop(scores_final.columns[[9,12,13]], axis=1, inplace=True)
# rename the columns
scores_final.columns = ["Exam Scores","Quiz Scores","Homework1 Scores","Homework2 Scores","Homework3 Scores","Homework4 Scores",]
# finally, concatenate scores_final into the panda_grades variable
panda_grades = panda_grades.drop(['scores'], axis=1)
panda_grades = pd.concat([panda_grades, scores_final], axis=1, join_axes=[panda_grades.index])
# let's drop the first column, since it's meaningless
panda_grades.drop(panda_grades.columns[[0]], axis=1, inplace=True)


# Questions
# 1. What is the average number of students in each class?
student_count = pd.DataFrame(panda_grades['class_id'].value_counts()) # count the # times class_id occurs, which equals the # of students
student_count = student_count.sort_index() # sort the index or class_id in ascending order
# note the "class_id" variable is really the count of the class_id in the panda_grades variable
student_count = student_count.reset_index().set_index('index', drop=False) # copy the index as a column so that it is plottable on the x-axis
student_count_plot = student_count.plot.bar(x="index",y="class_id",legend=False,title='# of Students in Each Classroom')
student_count_plot.set_xlabel("Classroom ID")
student_count_plot.set_ylabel("# of Students")
st.mean(student_count["class_id"]) # average number of students in each class is 9

# 2. What are the average scores across all classrooms?
# group by class_id
grouped = panda_grades.groupby('class_id')
exam_stats = grouped["Exam Scores"].agg([np.mean, np.std])
quiz_stats = grouped["Quiz Scores"].agg([np.mean, np.std])
hw1_stats = grouped["Homework1 Scores"].agg([np.mean, np.std])
hw2_stats = grouped["Homework2 Scores"].agg([np.mean, np.std])
hw3_stats = grouped["Homework3 Scores"].agg([np.mean, np.std])
hw4_stats = grouped["Homework4 Scores"].agg([np.mean, np.std])
# join them together into one dataframe
stats_all = pd.concat([exam_stats,quiz_stats,hw1_stats,hw2_stats,hw3_stats,hw4_stats], axis=1, join_axes=[exam_stats.index])
stats_all.columns = ["Avg. Exam Score","Std. Exam Score","Avg. Quiz Scores","Std. Quiz Scores","Avg. Homework1 Scores","Std. Homework1 Scores","Avg. Homework2 Scores","Std. Homework2 Scores","Avg. Homework3 Scores","Std. Homework3 Scores","Avg. Homework4 Scores","Std. Homework4 Scores"]
stats_all.reset_index(level=0, inplace=True)
avg_all = pd.concat([exam_stats["mean"],quiz_stats["mean"],hw1_stats["mean"],hw2_stats["mean"],hw3_stats["mean"],hw4_stats["mean"]], axis=1, join_axes=[exam_stats.index])
avg_all.columns = ["Avg. Exam Score","Avg. Quiz Scores","Avg. Homework1 Scores","Avg. Homework2 Scores","Avg. Homework3 Scores","Avg. Homework4 Scores",]
avg_all.plot.bar(rot=0) # plot all averages
avg_all.reset_index(level=0, inplace=True) # copy index/class_id and make it a column
# export statistics of all scores
stats_all.to_csv("stats_scores.csv", index=False)
# export average data scores
avg_all.to_csv("avg_scores.csv", index=False)

# plot all average scores by classroom
avg_exams = avg_all.plot.bar(x="class_id",y="Avg. Exam Score",legend=False,title='Avg. Exam Scores')
avg_quiz = avg_all.plot.bar(x="class_id",y="Avg. Quiz Scores",legend=False,title='Avg. Quiz Scores')
avg_hw1 = avg_all.plot.bar(x="class_id",y="Avg. Homework1 Scores",legend=False,title='Avg. HW1 Scores')
avg_hw2 = avg_all.plot.bar(x="class_id",y="Avg. Homework2 Scores",legend=False,title='Avg. HW2 Scores')
avg_hw3 = avg_all.plot.bar(x="class_id",y="Avg. Homework3 Scores",legend=False,title='Avg. HW3 Scores')
avg_hw4 = avg_all.plot.bar(x="class_id",y="Avg. Homework4 Scores",legend=False,title='Avg. HW4 Scores')

client.close()
