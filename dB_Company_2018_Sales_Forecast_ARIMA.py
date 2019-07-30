"""
This exercise seeks to forecast 2018 sales for "dB Company" and the country
in which the company resides, called "Country," using ARIMA models. ARIMA, or
autoregressive integrated moving average, models are constructed using a number
of techniques to identify certain elements of time series data, like seasonality,
in order to predict future values. The data are real-world business data, and proprietary
information is shielded. The time series data frequency is 52 weeks.
"""

import pandas as pd
from matplotlib import pyplot
from scipy import stats
import numpy as np
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
from pandas.plotting import autocorrelation_plot
from pmdarima import auto_arima # pip install pmdarima
from statsmodels.tsa.arima_model import ARIMA
from pandas import DataFrame

## DATA PREP
# The original datasets, which are real-world data, are not truly time series
# because they only show days on which a transaction was completed, so there are missing days.
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

# make index of 2018 sales DF Date column
db_sales_2018['Date'] = pd.to_datetime(db_sales_2018['Date'])
db_sales_2018 = db_sales_2018.set_index("Date")
Country_sales_2018['Date'] = pd.to_datetime(Country_sales_2018['Date'])
Country_sales_2018 = Country_sales_2018.set_index("Date")


## SEARCH FOR OUTLIERS
"""
After the data are imported, we identify and eliminate outliers in both the dB and Country sales datasets.
The idea is to remove outliers that will negatively influence the predictability of the model.
Note that we could also smooth the data using a moving average to reduce the volatility.
However, to encourage the most accurate predictions, each model will be constructed using outlier-reduced data instead.
An outlier is defined here as a value whose z-score of the data distribution is greater than 3. These outliers are then replaced by a NaN.
Finally, we replace the NaN with the average of the value before and after the NaN. Thus, the outliers are replaced by a local average.
Note that interpolation can also be used.
"""
db_sales.loc[stats.zscore(db_sales.Sales) > 3, 'Sales'] = np.nan # 3 is the typical threshold of stds away from the mean
db_sales.Sales = db_sales.Sales.fillna((db_sales.Sales.shift() + db_sales.Sales.shift(-1))/2)
Country_sales.loc[abs(stats.zscore(Country_sales.Sales) > 3), 'Sales'] = np.nan # 3 is the typical threshold of stds away from the mean
Country_sales.Sales = Country_sales.Sales.fillna((Country_sales.Sales.shift() + Country_sales.Sales.shift(-1))/2)
#Country_sales.iloc[[222,223],1] = 0 # remove extreme value missed in previous line
db_sales.plot()	
Country_sales.plot()


## DECOMPOSITION
"""
Decomposition extracts components such as seasonality and cyclicity from the data to help us determine the ARIMA parameters,
p (order of AR part), d (degree of differencing), and q (MA part). Specifically, if seasonality and trends are present, then we'll need to account
for them by identifying lags and/or by differencing. Decomposition is achieved using the mslt function, a multiplicative version of stl to allow for
multiple seasonal periods, outputs the time series into decomposed components.
Note: we will allow the ARIMA functions to implement differencing, if needed (e.g., parameter d in auto.arima() function).
This will save time since we will not need to integrate the forecasted data to retrieve the actual values of the forecast.
"""
db_decomp = db_sales # create date/time component required for TS analysis in Python
db_decomp['Date'] = pd.to_datetime(db_decomp['Date'])
db_decomp = db_decomp.set_index("Date") # set date as index for decomposition
res_db = sm.tsa.seasonal_decompose(db_decomp)
resplot_db = res_db.plot()

Country_decomp = Country_sales # create date/time component required for TS analysis in Python
Country_decomp['Date'] = pd.to_datetime(Country_decomp['Date'])
Country_decomp = Country_sales.set_index("Date") # set date as index for decomposition
res_Country = sm.tsa.seasonal_decompose(Country_decomp)
resplot_Country = res_Country.plot()


## STATIONARITY TEST
"""
Decomposition takes the first differences of the data to ensure that it is stationary. ARIMA doesn't require stationarity, but we still need to confirm
that the data are stationary or non-stationary after decomposition before proceding with the model. We invoke the Dickey-Fuller test to verify.
The null hypothesis is that the data are non-stationary, with an alpha value of 0.05 or 5%. The p-value needs to be less than or equal to 0.05 in order to
reject the null hypothesis, indicating we are at least 95% certain the alternative hypothesis, which is the data are stationary, is truer than the
null hypothesis. Viewing the results, we can safely reject the null hypothesis because the p-value is 0.01.
"""
print(sm.tsa.stattools.adfuller(db_decomp.Sales, maxlag=1, regression='ct', autolag='AIC', store=True, regresults=False))
print(sm.tsa.stattools.adfuller(Country_decomp.Sales, maxlag=1, regression='ct', autolag='AIC', store=True, regresults=False))


## ACF and PACF
"""
Let's look at the ACF and PACF of the data to identify lagged values, or lags. If a series' lags correlate or repeat, then there most likely is a
seasonal component to the series, which we'll want to account for. Identifying the lags helps to indicate which parameters (p, d, and/or q),
or even (P, D, and/or Q) for the season component we'll add to the ARIMA models, to adjust.
The ACF shows a spike at lag 1, and the PACF shows spikes around a number of lags, so this can be accounted for the in the models if necessary.
Otherwise, the peaks look relatively flat, and there's no obvious trend (e.g., sinusoidal characteristic).
"""
autocorrelation_plot(db_decomp)
pyplot.show()
autocorrelation_plot(Country_decomp)
pyplot.show()

## BUILDING ARIMA MODELS
"""
Now we can build an ARIMA model using the auto.arima function, which automatically "interprets" the data and assigned p, d, and q parameters.
Observing the ACF and PACF of the the auto ARIMA fit helps us to determine if the parameters need to be tweaked.
The auto.arima function shows fairly good ACF and PACF plots. However, none of the model's parameters are statistically significant;
thus, further adjustment of P, D, and Q parameters are required. In general, though it is not the case here, high peaks are usually indicators of
significant autocorrelation at those lags, which is something we must either eliminate or reduce as much as possible.
"""
# dB
db_model1 = auto_arima(db_decomp, seasonal=False, trace=True, stepwise=False)
print(db_model1.aic())
#db_model1.fit(disp=0)
db_forecast1 = db_model1.predict(n_periods=52)
# compare predicted and actual sales
db_forecast1 = pd.DataFrame(db_forecast1,index = db_sales_2018.index,columns=['Prediction'])
pd.concat([db_sales_2018,db_forecast1],axis=1).plot()
# this inadequate prediction indicates we'll need to adjust parameters for seasonality.
# build another model using ARIMA with seasonality
db_model2 = sm.tsa.statespace.SARIMAX(db_decomp, order=(4,0,0),seasonal_order=(1,1,0,12)).fit() # this function allows us to add a seasonal component
db_forecast2 = db_model2.predict(start="2017-12-24",end="2018-12-30")
# compare predicted and actual sales
db_forecast2 = pd.DataFrame(db_forecast2,index = db_sales_2018.index,columns=['Prediction'])
pd.concat([db_sales_2018,db_forecast2],axis=1).plot()

# Country
Country_model1 = auto_arima(Country_decomp, seasonal=False, trace=True, stepwise=False)
print(Country_model1.aic())
#db_model1.fit(disp=0)
Country_forecast1 = Country_model1.predict(n_periods=52)
# compare predicted and actual sales
Country_forecast1 = pd.DataFrame(Country_forecast1,index = Country_sales_2018.index,columns=['Prediction'])
pd.concat([Country_sales_2018,Country_forecast1],axis=1).plot()
# this inadequate prediction indicates we'll need to adjust parameters for seasonality.
# build another model using ARIMA with seasonality
Country_model2 = sm.tsa.statespace.SARIMAX(Country_decomp, order=(1,0,0),seasonal_order=(1,1,0,12)).fit() # this function allows us to add a seasonal component
Country_forecast2 = Country_model2.predict(start="2017-12-24",end="2018-12-30")
# compare predicted and actual sales
Country_forecast2 = pd.DataFrame(Country_forecast2,index = Country_sales_2018.index,columns=['Prediction'])
pd.concat([Country_sales_2018,Country_forecast2],axis=1).plot()

"""
The forecasted traces, neither from the auto AIRMA nor seasonal ARIMA models, fit the predicted values exactly. On the plus side, the sum predicted totals
are not too far off from the actual sales values:
    dB predicted: 2,282,118 euros
    dB actual: 1,829,933 euros
    Country predicted: 3,993,842 euros
    Country actual: 3,960,435 euros
"""