# -*- coding: utf-8 -*-
"""
DSC 630
Final

David Brehm
"""

#%%
import pandas as pd
import matplotlib.pyplot as plt
from functools import reduce
import glob
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
import datetime as dt
import numpy as np

#%%
# Set data filepath and get files.
fp = r'D:\School\630\Data\Final'
files = glob.glob(fp + "/*.csv")

df_list = []
for fn in files:
    df_temp = pd.read_csv(fn)
    df_list.append(df_temp)
   
df = reduce(lambda left, right: pd.merge(left,right, on='DATE', 
                                         how='outer'), df_list)

#%%
# Quick report on data.
prof = ProfileReport(df)
prof.to_file(output_file=r'D:\School\630\Data\Final\output.html')

#%%
# Fill in data gaps with interpolation. Subset the data by date range.
df['MEPAINUSA672N'] = df['MEPAINUSA672N'].interpolate()
df['GDI'] = df['GDI'].interpolate()

df = df[(df['DATE'] >= '2003-01-01') & (df['DATE'] < '2021-05-01')]

#%%
# Convert date to ordinal.
df['DATE'] = pd.to_datetime(df['DATE'])
x_date = df['DATE']
df['DATE'] = df['DATE'].map(dt.datetime.toordinal)

#%%
# Split into independent and dependent variables.
X = df.drop('RSGCS', axis=1)
y = df['RSGCS']

#%%
# Train and test over entire dataset.
poly = PolynomialFeatures(degree=2)
poly_var = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(poly_var, y, test_size=0.11, random_state=0)


linreg = linear_model.LinearRegression()

mod = linreg.fit(X_train, y_train)
score = mod.score(X_test, y_test)

y_pred = mod.predict(X_train)
X_date = pd.Series(X_train[:,1].astype(int)).apply(dt.datetime.fromordinal)

plt.scatter(X_date, y_train, color='blue', s=8, label='Actual')
plt.scatter(X_date, y_pred, color='red', s=5, label='Predicted')
plt.title('US Grocery Store Sales')
plt.xlabel('Year')
plt.ylabel('Millions of Dollars')
plt.legend()

plt.savefig(r'D:\School\630\milestone4_5.png')


#%%
# Train on beginning of dataset, forecast end of dataset.
poly = PolynomialFeatures(degree=2)
poly_var = poly.fit_transform(X)

X_train1, X_test1 = np.split(poly_var, [int(.89 *len(poly_var))])
y_train1, y_test1 = np.split(y, [int(.89 *len(y))])

linreg = linear_model.LinearRegression()

mod = linreg.fit(X_train1, y_train1)
score = mod.score(X_test1, y_test1)

y_pred1 = mod.predict(X_train1)
X_date1 = pd.Series(X_train1[:,1].astype(int)).apply(dt.datetime.fromordinal)

y_fore = mod.predict(X_test1)
X_date2 = pd.Series(X_test1[:,1].astype(int)).apply(dt.datetime.fromordinal)


plt.plot(x_date, y, color='blue', label='Actual')
plt.plot(X_date1, y_pred1, color='red', label='Predicted')
plt.plot(X_date2, y_fore, color='orange', label='Forecast')
plt.title('US Grocery Store Sales')
plt.xlabel('Year')
plt.ylabel('Millions of Dollars')
plt.legend()

plt.savefig(r'D:\School\630\milestone5_1.png')
