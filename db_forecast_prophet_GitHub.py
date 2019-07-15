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

# build the forecasting model using Prophet
db_sales.columns = ["ds","y"] # Prophet requires these column names
Country_sales.columns = ["ds","y"] # Prophet requires these column names
mdb = Prophet()
mCountry = Prophet()
mdb.fit(db_sales)
mCountry.fit(Country_sales)
# forecast future weekly values for 2019
future_db = mdb.make_future_dataframe(periods=52)
future_Country = mCountry.make_future_dataframe(periods=52)
future_db.tail()
future_Country.tail()
# determine upper and lower prediction bounds
forecast_db = mdb.predict(future_db)
forecast_db[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast_Country = mCountry.predict(future_Country)
forecast_Country[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# plot
fig_db1 = mdb.plot(forecast_db)
fig_db2 = mdb.plot_components(forecast_db)
fig_Country1 = mdb.plot(forecast_Country)
fig_Country2 = mdb.plot_components(forecast_Country)



# PART 2
# 2018 prediction
db_sales_2018 = db_sales[0:208]
Country_sales_2018 = Country_sales[0:208]

# we don't need to worry about all the formatting and date/time tweaks made above, since we're working with the adjusted dataset

# build the forecasting model using Prophet
db_sales_2018.columns = ["ds","y"] # Prophet requires these column names
Country_sales_2018.columns = ["ds","y"] # Prophet requires these column names
mdb = Prophet()
mCountry = Prophet()
mdb.fit(db_sales_2018)
mCountry.fit(Country_sales_2018)
# forecast future weekly values for 2018
future_db_2018 = mdb.make_future_dataframe(periods=52)
future_Country_2018 = mCountry.make_future_dataframe(periods=52)
future_db_2018.tail()
future_Country_2018.tail()
# determine upper and lower prediction bounds
forecast_db_2018 = mdb.predict(future_db_2018)
forecast_db_2018[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast_Country_2018 = mCountry.predict(future_Country_2018)
forecast_Country_2018[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# plot
fig_db1 = mdb.plot(forecast_db_2018)
fig_db2 = mdb.plot_components(forecast_db_2018)
fig_Country1 = mdb.plot(forecast_Country_2018)
fig_Country = mdb.plot_components(forecast_Country_2018)

"""
2018 sales were $1,832,070. Prophet forecast 2018 sales to be $1,840,778 (accurate to 99.5%), which indicates Prophet was able to model
the historical data accurately, increasing our confidence in the 2019 sales prediction of $2,059,648.
"""



"""
Use of FB's Prophet for sales forecasting. In Python, PyStan is this package's dependency. Install instructions: https://facebook.github.io/prophet/docs/installation.html

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
"""
