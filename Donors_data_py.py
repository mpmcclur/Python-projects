"""
Matt McClure
Exploring the Donors dataset using Pandas.
Basic data manipulation and statistical and graphical information regarding the dataset are applied.
"""

import pandas as pd
import matplotlib.pyplot as pp
import statistics as st

donors = pd.read_csv("C:/Users/Matt/Desktop/IST 652 (Scripting for Data Analysis)/Homework 1/donors_data.csv") # read in our data
donors = donors.set_index('Row Id')

# Data Cleaning and Prep
# reassign unique values to zipconvert columns (e.g., values of 1 in zipconvert_3 will be 2, zipconvert_4 will be 3, etc.)
donors.zipconvert_3 = donors.zipconvert_3.replace({1:2})
donors.zipconvert_4 = donors.zipconvert_4.replace({1:3})
donors.zipconvert_5 = donors.zipconvert_5.replace({1:4})
# combine zip code columns into one zipconvert variable called zipconvert0
donors["zipconvert0"] = donors["zipconvert_2"] + donors["zipconvert_3"] + donors["zipconvert_4"]+ donors["zipconvert_5"]
donors['zipconvert_2'] = donors['zipconvert0'] # replace zipconvert_2 values with new zipconvert0 values
donors.rename(columns = {'zipconvert_2':'Region'}, inplace = True) # rename zipconvert_2 to Region
# delete Row Id. and old zipconvert variables
del donors["Row Id."]
del donors["zipconvert_3"]
del donors["zipconvert_4"]
del donors["zipconvert_5"]
del donors["zipconvert0"]
# replace rest of column names
donors.rename(columns={'homeowner dummy': 'Homeowner', 'NUMCHLD': 'Number of Children', 'INCOME': 'Income', 'gender dummy': 'Gender', 'WEALTH': 'Wealth Rating', 'HV': 'Avg. Home Value', 'Icmed': 'Median Family Income', 'Icavg': 'Avg. Family Income', 'IC15': '% Low Income', 'NUMPROM': '# of Promotions', 'RAMNTALL': 'Total Gifts Amount', 'MAXRAMNT': 'Max Gift Amount', 'LASTGIFT': 'Last Gift Amount', 'totalmonths': 'Months Since Last Donation', 'TIMELAG': 'Months b/w 1st and 2nd Gift', 'AVGGIFT': 'Avg. Gift Amount', 'TARGET_B': 'Donor?', 'TARGET_D': 'Predicted Gift Amount'}, inplace=True)

# Data Analysis
# Let's compare donors in the various zip codes with various types or amounts of giving.
# Create separate variable to view other variables by region
donors_region1 = donors.loc[donors['Region'] == 1]
donors_region2 = donors.loc[donors['Region'] == 2]
donors_region3 = donors.loc[donors['Region'] == 3]
donors_region4 = donors.loc[donors['Region'] == 4]
len(donors_region1) # 669 households
len(donors_region2) # 578 households
len(donors_region3) # 669 households
len(donors_region4) # 1200 households
# How many donors are there in each region?
sum(donors_region1["Donor?"]) # 337
sum(donors_region2["Donor?"]) # 281
sum(donors_region3["Donor?"]) # 326
sum(donors_region4["Donor?"]) # 616; region 4 has the highest number of donors by a large margin
sum_donors = {"Region 1":sum(donors_region1["Donor?"]),"Region 2":sum(donors_region2["Donor?"]),"Region 3":sum(donors_region3["Donor?"]),"Region 4":sum(donors_region4["Donor?"])}
# Let's plot this:
pp.bar(sum_donors.keys(), sum_donors.values(), color='g')
pp.title("# of Donors in Each Region")
pp.show()
# What is the mode of income rating in each region?
wealth_rating = {"Region 1":st.mode(donors_region1["Wealth Rating"]), "Region 2" :st.mode(donors_region2["Wealth Rating"]), "Region 3":st.mode(donors_region3["Wealth Rating"]), "Region 4":st.mode(donors_region4["Wealth Rating"])}
pp.bar(wealth_rating.keys(), wealth_rating.values(), color='r')
pp.title("Most Common Wealth Rating in Each Region")
pp.show()
# What is the sum of the total gifts amount in each region?
regions_sum = {"Region 1":sum(donors_region1["Total Gifts Amount"]), "Region 2" :sum(donors_region2["Total Gifts Amount"]), "Region 3":sum(donors_region3["Total Gifts Amount"]), "Region 4":sum(donors_region4["Total Gifts Amount"])}
pp.bar(regions_sum.keys(), regions_sum.values(), color='b')
pp.title("Total # of Gifts in Each Region")
pp.show()
# What is the average number of promotions in each region?
avg_promotions = {"Region 1":st.mean(donors_region1["# of Promotions"]), "Region 2" :st.mean(donors_region2["# of Promotions"]), "Region 3":st.mean(donors_region3["# of Promotions"]), "Region 4":st.mean(donors_region4["# of Promotions"])}
pp.bar(avg_promotions.keys(), avg_promotions.values(), color='y')
pp.title("Most Common Wealth Rating in Each Region")
pp.show()
# Output the files as CSVs:
donors[donors["Region"] == 1].to_csv("region1_donors.csv", index=False)
donors[donors["Region"] == 2].to_csv("region2_donors.csv", index=False)
donors[donors["Region"] == 3].to_csv("region3_donors.csv", index=False)
donors[donors["Region"] == 4].to_csv("region4_donors.csv", index=False)

# Considerations for other analyses
# Let's strip the data to compare three variables: the number of promotions with the total amount of donations and the frequency of donations.
# Output promotions that were <50
donors[donors["# of Promotions"] < 50].to_csv("less_than_50_promotions.csv",columns=['# of Promotions', 'Total Gifts Amount', 'Months Since Last Donation'], index=False)
# Output promotions that were >= 50
donors[donors["# of Promotions"] >= 50].to_csv("at_least_50_promotions.csv",columns=['# of Promotions', 'Total Gifts Amount', 'Months Since Last Donation'], index=False)

# Let's strip the data to compare the number of months since the last donation to the donation amounts.
donors.to_csv("last_donation_amounts.csv",columns=['Months Since Last Donation', 'Total Gifts Amount'], index=False)
