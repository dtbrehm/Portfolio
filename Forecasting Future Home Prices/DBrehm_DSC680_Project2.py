# -*- coding: utf-8 -*-
"""
Forecasting Home Prices.

@author: David Brehm
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.holtwinters import ExponentialSmoothing

#%% Get data.
datapath = r'D:\School\680\Project 2\Data\Zip_zhvi_uc_sfr_tier_0.33_0.67_sm_sa_month.csv'

housezip = pd.read_csv(datapath)
housezip2 = housezip.copy()

#%% Get percent change for all zip codes.
housezip2['per_change'] = (housezip2['2021-08-31'] - housezip2['2000-01-31'])/housezip2['2000-01-31'] * 100


#%% Get median home values for all zip codes and top/bottom decile.
all_med = housezip.median(axis=0).iloc[3:].reset_index()
all_med['index'] = pd.to_datetime(all_med['index'], format='%Y-%m-%d')

top_dec = housezip[housezip['2021-08-31'] >= np.percentile(housezip['2021-08-31'],90)]
top_med = top_dec.median(axis=0).iloc[3:].reset_index()
top_med['index'] = pd.to_datetime(top_med['index'], format='%Y-%m-%d')
top_dec['per_change'] = (top_dec['2021-08-31'] - top_dec['2000-01-31'])/top_dec['2000-01-31'] * 100


bot_dec = housezip[housezip['2021-08-31'] <= np.percentile(housezip['2021-08-31'],10)]
bot_med = bot_dec.median(axis=0).iloc[3:].reset_index()
bot_med['index'] = pd.to_datetime(bot_med['index'], format='%Y-%m-%d')
bot_dec['per_change'] = (bot_dec['2021-08-31'] - bot_dec['2000-01-31'])/bot_dec['2000-01-31'] * 100

#%% Plot median home value for all zip codes, top decile of current values, and bottom decile of current values
plt.plot(all_med['index'], all_med[0], label="All Zip Codes")
plt.plot(top_med['index'], top_med[0], label="Top Decile Current Value")
plt.plot(bot_med['index'], bot_med[0], label="Bottom Decile Current Value")
plt.legend(loc="upper left", prop={'size': 9})
plt.xlabel('Date')
plt.ylabel('Home Value ($)')
plt.title('Median Home Values by Zip Code')

#%% Look at percent change since 2000.
print('Median percent change, all zip codes: '.ljust(60),
      round(np.nanpercentile(housezip2['per_change'],50),2), '%')
print('Median percent change, top decile current home value: '.ljust(60), 
      round(np.nanpercentile(top_dec['per_change'],50),2), '%')
print('Median percent change, bottom decile current home value: '.ljust(60), 
      round(np.nanpercentile(bot_dec['per_change'],50),2), '%')


#%% Split data into train and test sets. Test dataset is the last three years.
train_all = all_med[:224]
test_all = all_med[224:]
train_all.index = train_all['index']
test_all.index = test_all['index']   

train_top = top_med[:224]
test_top = top_med[224:]
train_top.index = train_top['index']
test_top.index = test_top['index']

train_bot = bot_med[:224]
test_bot = bot_med[224:]
train_bot.index = train_bot['index']
test_bot.index = test_bot['index']   

#%% Exponential Smoothing model.
modelES_all = ExponentialSmoothing(np.asarray(train_all[0].values), seasonal_periods=12, 
                             trend='add', seasonal='add',).fit()
modelES_all._index = train_all.index
predES_all = modelES_all.forecast(36)

modelES_top = ExponentialSmoothing(np.asarray(train_top[0].values), seasonal_periods=12, 
                             trend='add', seasonal='add',).fit()
modelES_top._index = train_top.index
predES_top = modelES_top.forecast(36)

modelES_bot = ExponentialSmoothing(np.asarray(train_bot[0].values), seasonal_periods=12, 
                             trend='add', seasonal='add',).fit()
modelES_bot._index = train_bot.index
predES_bot = modelES_bot.forecast(36)

#%% Plot Exponential Smoothing for all zip codes.
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_all.index, train_all[0], color='blue', linewidth=3, label='Train Data')
ax.plot(test_all.index, test_all[0], color="gray", linewidth=3, label='Test Data')
ax.plot(train_all.index, modelES_all.fittedvalues, color='red', label='Exponential Smoothing Model')
ax.plot(test_all.index, predES_all, color='red',linestyle='dashed', label='Exponential Smoothing Forecast')
ax.vlines(x=pd.to_datetime('2018-09-15T00:00:00.000000000'),ymin=0, ymax=200000, colors='black')
plt.legend(loc=3)
plt.xlabel('Date')
plt.ylabel('Home Value ($)')
plt.title('Forecasting All Home Values by Exponential Smoothing')

#%% Plot Exponential Smoothing for top decile of current home value zip codes.
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_top.index, train_top[0], color='blue', linewidth=3, label='Train Data')
ax.plot(test_top.index, test_top[0], color="gray", linewidth=3, label='Test Data')
ax.plot(train_top.index, modelES_top.fittedvalues, color='red', label='Exponential Smoothing Model')
ax.plot(test_top.index, predES_top, color='red',linestyle='dashed', label='Exponential Smoothing Forecast')
ax.vlines(x=pd.to_datetime('2018-09-15T00:00:00.000000000'),ymin=0, ymax=800000, colors='black')
plt.legend(loc=3)
plt.xlabel('Date')
plt.ylabel('Home Value ($)')
plt.title('Forecasting Top Decile Home Values by Exponential Smoothing')

#%% Plot Exponential Smoothing for bottom decile of current home value zip codes.
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(train_bot.index, train_bot[0], color='blue', linewidth=3, label='Train Data')
ax.plot(test_bot.index, test_bot[0], color="gray", linewidth=3, label='Test Data')
ax.plot(train_bot.index, modelES_bot.fittedvalues, color='red', label='Exponential Smoothing Model')
ax.plot(test_bot.index, predES_bot, color='red',linestyle='dashed', label='Exponential Smoothing Forecast')
ax.vlines(x=pd.to_datetime('2018-09-15T00:00:00.000000000'),ymin=0, ymax=75000, colors='black')
plt.legend(loc=3)
plt.xlabel('Date')
plt.ylabel('Home Value ($)')
plt.title('Forecasting Bottom Decile Home Values by Exponential Smoothing')

#%% Moving Average, window of three.

window = 3
hist = [all_med[0][i] for i in range(window)]
test_MA_all = [all_med[0][i] for i in range(window, len(all_med))]
pred_MA_all = []
for j in range(len(test_MA_all)):
    length = len(hist)
    y_MA_all = np.mean([hist[i] for i in range(length-window, length)])
    obs = test_MA_all[j]
    pred_MA_all.append(y_MA_all)
    hist.append(obs)

#%% Plot moving average.
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(all_med['index'][3:], test_MA_all, color='blue', linewidth=3, label='Home Values')
ax.plot(all_med['index'][3:], pred_MA_all, color='red', label='Moving Average')
plt.legend(loc=2)
plt.xlabel('Date')
plt.ylabel('Home Value ($)')
plt.title('Predicting Home Values by Moving Average')

#%% Get MSE for the models.
print('')
print('Exponential Smoothing MSE, All Zip Codes : '.ljust(50), round(mean_squared_error(test_all[0], predES_all),2))
print('Exponential Smoothing MSE, Top Decile : '.ljust(50), round(mean_squared_error(test_top[0], predES_top),2))
print('Exponential Smoothing MSE, Bottom Decile : '.ljust(50), round(mean_squared_error(test_bot[0], predES_bot),2))
print('Moving Average MSE: '.ljust(50), round(mean_squared_error(test_MA_all, pred_MA_all),2))
