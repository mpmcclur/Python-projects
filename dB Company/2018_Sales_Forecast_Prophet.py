"""
This program is a quick exercise to test the accuracy of Facebook's Prophet algorithm, purportedly based on Bayesian model fitting.
The dataset used is a real dataset; the goal is to predict 2019 sales of both "dB Company" and sales from "Country".
Part 1 seeks to create the model, and Part 2 tests the model to predict 2018 data, which exists. The model was able to predict 2018 sales data to 99.5% accuracy.
Compare this result with the ARIMA model constructed in R in my R-projects folder on GitHub.
"""
import pandas as pd
from fbprophet import Prophet


# PART 1
db_sales = pd.read_excel('dB_Sales.xlsx',sheet_name='dB_Sales')
Country_sales = pd.read_excel('dB_Sales.xlsx',sheet_name='all_Country')

# replace column names
db_sales.columns = ["Date","Sales"]
Country_sales.columns = ["Date","Sales"]
db_sales.Date = pd.to_datetime(db_sales.Date,format='%m-%d-%Y')
Country_sales.Date = pd.to_datetime(Country_sales.Date,format='%m-%d-%Y')

# fill in missing days in both datasets
r = pd.date_range(start=min(db_sales.Date),end=max(db_sales.Date)) # set daily range for dB
r2 = pd.date_range(start=min(Country_sales.Date),end=max(Country_sales.Date)) # set daily range for Country
db_sales = db_sales.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()
Country_sales = Country_sales.set_index('Date').reindex(r2).fillna(0.0).rename_axis('Date').reset_index()

# convert daily data into weekly data
db_sales = db_sales.set_index('Date') # must convert Date column to index
Country_sales = Country_sales.set_index('Date') # must convert Date column to index
db_sales = db_sales.resample('W').agg({'Sales':sum})
Country_sales = Country_sales.resample('W').agg({'Sales':sum})

# drop time component from Date column
db_sales = db_sales.reset_index(drop=False) # reset index and make Date a column again
Country_sales = Country_sales.reset_index(drop=False) # reset index and make Date a column again
db_sales.Date = db_sales.Date.dt.date
Country_sales.Date = Country_sales.Date.dt.date
db_sales.plot()	
Country_sales.plot()
db_sales_2018 = db_sales.iloc[209:261,:] # split 2018 sales into new DF
Country_sales_2018 = Country_sales[208:260]
db_sales = db_sales[0:208] # create DF of 2014-2017 data
Country_sales = Country_sales[0:208] # create DF of 2014-2017 data

# reset index of 2018 data
db_sales_2018 = db_sales_2018.reset_index(drop=True)
Country_sales_2018 = Country_sales_2018.reset_index(drop=True)



# PART 2
# we don't need to worry about all the formatting and date/time tweaks made above, since we're working with the adjusted dataset
# build the forecasting model using Prophet
db_sales.columns = ["ds","y"] # Prophet requires these column names
Country_sales.columns = ["ds","y"] # Prophet requires these column names
mdb = Prophet()
mCountry = Prophet()
mdb.fit(db_sales)
mCountry.fit(Country_sales)
# forecast future weekly values for 2018
future_db = mdb.make_future_dataframe(periods=52)
future_Country = mCountry.make_future_dataframe(periods=52)
future_db.tail()
future_Country.tail()
# determine upper and lower prediction bounds
forecast_db = mdb.predict(future_db)
forecast_db[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# create new df of forecasted 2018 sales
forecast_db_2018 = forecast_db.iloc[208:]
# reset index
forecast_db_2018 = forecast_db_2018.reset_index(drop=True)
forecast_Country = mCountry.predict(future_Country)
forecast_Country[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# create new df of forecasted 2018 sales
forecast_Country_2018 = forecast_Country.iloc[208:]
forecast_Country_2018 = forecast_Country_2018.reset_index(drop=True)
# reset index
forecast_Country_2018 = forecast_Country_2018.reset_index(drop=True)
# plot
fig_db1 = mdb.plot(forecast_db)
fig_db2 = mdb.plot_components(forecast_db)
fig_Country1 = mdb.plot(forecast_Country)
fig_Country = mdb.plot_components(forecast_Country)

# compare predicted versus actual
db_2018_compare = pd.DataFrame(forecast_db,index = db_sales_2018.index,columns=['yhat'])
pd.concat([db_sales_2018,db_2018_compare],axis=1).plot()

Country_2018_compare = pd.DataFrame(forecast_Country,index = Country_sales_2018.index,columns=['yhat'])
pd.concat([Country_sales_2018,Country_2018_compare],axis=1).plot()

"""
Prophet was unable to forecast the 2018 sales trends, but both models were able to capture a minor trend found in the actual 2018 sales.
Even though the models couldn't accurately predict the peaks, the predicted total for both dB and Country are close to the actual total.
Below are the results:
    dB predicted: 1,955,652 euros
    dB actual: 1,829,933 euros
    Country predicted: 3,815,905 euros
    Country actual: 3,960,435 euros
"""



"""
Use of FB's Prophet for sales forecasting. In Python, PyStan is this package's dependency.
Install instructions: https://facebook.github.io/prophet/docs/installation.html

In conda shell, execute the following:
conda install libpython m2w64-toolchain -c msys2
conda install numpy cython -c conda-forge
conda install matplotlib scipy pandas -c conda-forge
pip install pystan
conda install pystan -c conda-forge
conda install numpy cython matplotlib scipy pandas -c conda-forge
pip install ephem
conda install numpy
pip install fbprophet
conda install plotly
conda install matplotlib

https://facebook.github.io/prophet/docs/quick_start.html

# compare predicted versus actual
# make index of 2018 sales DF Date column
db_sales_2018['Date'] = pd.to_datetime(db_sales_2018['Date'])
db_sales_2018 = db_sales_2018.set_index("Date")
Country_sales_2018['Date'] = pd.to_datetime(Country_sales_2018['Date'])
Country_sales_2018 = Country_sales_2018.set_index("Date")

forecast_db['ds'] = pd.to_datetime(forecast_db['ds'])
forecast_db = forecast_db.set_index("ds")

forecast_Country['ds'] = pd.to_datetime(forecast_Country['ds'])
forecast_Country = forecast_Country.set_index("ds")
"""
