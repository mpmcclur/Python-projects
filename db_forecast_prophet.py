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

import pandas as pd
from fbprophet import Prophet

db_sales = pd.read_excel('dB_Sales.xlsx',sheet_name='dB_Sales')
Italy_sales = pd.read_excel('dB_Sales.xlsx',sheet_name='all_Italy')

# replace column names
db_sales.columns = ["Date","Sales"]
Italy_sales.columns = ["Date","Sales"]
db_sales.Date = pd.to_datetime(db_sales.Date,format='%m-%d-%Y')
Italy_sales.Date = pd.to_datetime(Italy_sales.Date,format='%m-%d-%Y')

# fill in missing days in both datasets
r = pd.date_range(start=min(db_sales.Date),end=max(db_sales.Date)) # set daily range for dB
r2 = pd.date_range(start=min(Italy_sales.Date),end=max(Italy_sales.Date)) # set daily range for Italy
db_sales = db_sales.set_index('Date').reindex(r).fillna(0.0).rename_axis('Date').reset_index()
Italy_sales = Italy_sales.set_index('Date').reindex(r2).fillna(0.0).rename_axis('Date').reset_index()

# convert daily data into weekly data
db_sales = db_sales.set_index('Date') # must convert Date column to index
Italy_sales = Italy_sales.set_index('Date') # must convert Date column to index
db_sales = db_sales.resample('W').agg({'Sales':sum})
Italy_sales = Italy_sales.resample('W').agg({'Sales':sum})

# drop time component from Date column
db_sales = db_sales.reset_index(drop=False) # reset index and make Date a column again
Italy_sales = Italy_sales.reset_index(drop=False) # reset index and make Date a column again
db_sales.Date = db_sales.Date.dt.date
Italy_sales.Date = Italy_sales.Date.dt.date

# build the forecasting model using Prophet
db_sales.columns = ["ds","y"] # Prophet requires these column names
Italy_sales.columns = ["ds","y"] # Prophet requires these column names
mdb = Prophet()
mItaly = Prophet()
mdb.fit(db_sales)
mItaly.fit(Italy_sales)
# forecast future weekly values for 2019
future_db = mdb.make_future_dataframe(periods=52)
future_Italy = mItaly.make_future_dataframe(periods=52)
future_db.tail()
future_Italy.tail()
# determine upper and lower prediction bounds
forecast_db = mdb.predict(future_db)
forecast_db[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast_Italy = mItaly.predict(future_Italy)
forecast_Italy[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# plot
fig_db1 = mdb.plot(forecast_db)
fig_db2 = mdb.plot_components(forecast_db)
fig_Italy1 = mdb.plot(forecast_Italy)
fig_Ital2 = mdb.plot_components(forecast_Italy)



# 2018 prediction
db_sales_2018 = db_sales[0:208]
Italy_sales_2018 = Italy_sales[0:208]

# we don't need to worry about all the formatting and date/time tweaks made above, since we're working with the adjusted dataset

# build the forecasting model using Prophet
db_sales_2018.columns = ["ds","y"] # Prophet requires these column names
Italy_sales_2018.columns = ["ds","y"] # Prophet requires these column names
mdb = Prophet()
mItaly = Prophet()
mdb.fit(db_sales_2018)
mItaly.fit(Italy_sales_2018)
# forecast future weekly values for 2018
future_db_2018 = mdb.make_future_dataframe(periods=52)
future_Italy_2018 = mItaly.make_future_dataframe(periods=52)
future_db_2018.tail()
future_Italy_2018.tail()
# determine upper and lower prediction bounds
forecast_db_2018 = mdb.predict(future_db_2018)
forecast_db_2018[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
forecast_Italy_2018 = mItaly.predict(future_Italy_2018)
forecast_Italy_2018[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# plot
fig_db1 = mdb.plot(forecast_db_2018)
fig_db2 = mdb.plot_components(forecast_db_2018)
fig_Italy1 = mdb.plot(forecast_Italy_2018)
fig_Ital2 = mdb.plot_components(forecast_Italy_2018)

"""
2018 sales were $1,832,070. Prophet forecast 2018 sales to be $1,840,778 (accurate to 99.5%), which indicates Prophet was able to model
the historical data accurately, increasing our confidence in the 2019 sales prediction of $2,059,648.
"""