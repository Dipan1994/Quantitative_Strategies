#!/usr/bin/env python
# coding: utf-8

# In[37]:


import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.tsa.stattools as ts
import statsmodels.api as sm
import matplotlib.pyplot as plt


# In[2]:


start_date = "2010-8-01"
end_date = "2016-08-01"    

prices = yf.download('GOOG', start=start_date, end=end_date)['Adj Close']


# In[4]:


#Test #1 - ADF test
ts.adfuller(prices, 1)
#if test test statistic is greater than critical value, not a mean-reverting process (may be a random walk)


# In[18]:


#Test #2 - Hurst exponent
#H<0.5 Mean reverting; H=0.5 Geometric Brownian Motion; H>0.5 Trending
lags = range(2, 100)
tau = [np.sqrt(np.std(np.subtract(np.array(prices[lag:]), np.array(prices[:-lag])))) for lag in lags]
poly = np.polyfit(np.log(lags), np.log(tau), 1)
poly[0]*2.0


# In[33]:


###With Exxon Mobil (XOM) and United States Oil Fund (USO)
start_date = '2019-01-01'
end_date = '2020-01-01'
prices = yf.download(['XOM','USO'], start=start_date, end=end_date)['Adj Close']


# In[34]:


price_data = [prices["USO"], prices["XOM"]]
headers = ["USO Price($)", "XOM Price($)"]
price_df = pd.concat(price_data, axis=1, keys=headers)


# In[35]:


fig = price_df.plot(title="USO and XOM Daily Prices")
fig.set_ylabel("Price($)")
plt.show()


# In[36]:


price_df.plot.scatter(x=0, y=1, title="USO and XOM Price Scatterplot")


# In[41]:


Y = price_df['USO Price($)']
x = price_df['XOM Price($)']
x = sm.add_constant(x)
model = sm.OLS(Y, x)
res = model.fit()

beta_hr = res.params[1]
print(f'Beta Hedge Ratio: {beta_hr}')
    
price_df["Residuals"] = res.resid

ts.adfuller(price_df["Residuals"])
#test statistics significant at 5% - indicating a cointegrating relationship


# In[42]:


#Pairs Strategy using mean reversion
import matplotlib.pyplot as plt
import numpy as np
import os, os.path
import pandas as pd
import seaborn as sns

import statsmodels.api as sm
from statsmodels.regression.rolling import RollingOLS

sns.set_style("darkgrid")


# In[49]:


pairs = yf.download(tickers=["SPY",'IWM'], period="7d", interval="1m")['Adj Close']
pairs = pairs.dropna()


# In[61]:


#Lookback Regression
lookback = 100
model = RollingOLS(
        endog=pairs['SPY'],
        exog=sm.add_constant(pairs['IWM']),
        window=lookback
    )
rres = model.fit()
params = rres.params.copy()
pairs['hedge_ratio'] = params['IWM']
pairs.dropna(inplace=True,axis=0)


# In[63]:


pairs['spread'] = (pairs['SPY'] - pairs['hedge_ratio']*pairs['IWM'])
pairs['zscore'] = (pairs['spread'] - np.mean(pairs['spread']))/np.std(pairs['spread'])


# In[71]:


#Long Short Strategy
z_entry_threshold=1
z_exit_threshold=0.5
pairs['longs'] = (pairs['zscore'] <= -z_entry_threshold)*1.0
pairs['shorts'] = (pairs['zscore'] >= z_entry_threshold)*1.0
pairs['exits'] = (np.abs(pairs['zscore']) <= z_exit_threshold)*1.0


# In[74]:


pairs['long_market'] = 0.0
pairs['short_market'] = 0.0

long_market = 0
short_market = 0

for i, b in enumerate(pairs.iterrows()):
    if b[1]['longs'] == 1.0:
        long_market = 1            
    if b[1]['shorts'] == 1.0:
        short_market = 1
    if b[1]['exits'] == 1.0:
        long_market = 0
        short_market = 0
    pairs.iloc[i]['long_market'] = long_market
    pairs.iloc[i]['short_market'] = short_market


# In[75]:


#portfolio returns
portfolio = pd.DataFrame(index=pairs.index)
portfolio['positions'] = pairs['long_market'] - pairs['short_market']
portfolio['SPY'] = -1.0 * pairs['SPY'] * portfolio['positions']
portfolio['IWM'] = pairs['IWM'] * portfolio['positions']
portfolio['total'] = portfolio['SPY'] + portfolio['IWM']

portfolio['returns'] = portfolio['total'].pct_change()
portfolio['returns'].fillna(0.0, inplace=True)
portfolio['returns'].replace([np.inf, -np.inf], 0.0, inplace=True)
portfolio['returns'].replace(-1.0, 0.0, inplace=True)

(portfolio['returns'] + 1.0).cumprod()[-1]


# In[77]:


fig = plt.figure()

ax1 = fig.add_subplot(211,  ylabel='SPY')
(pairs['SPY'].pct_change()+1.0).cumprod().plot(ax=ax1, color='r', lw=2.)

ax2 = fig.add_subplot(212, ylabel='Portfolio value growth (%%)')
(portfolio['returns'] + 1.0).cumprod().plot(ax=ax2, lw=2.)


# In[ ]:




