# Predicting US Grocery Store Sales

-----------
### Table of Contents

[Introduction](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#introduction)

[Conclusion](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#conclusion)

[References](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#references)

-----------
## Introduction
The recent pandemic has had an effect on many parts of our country, with the economy being one of the clearer ones. The shutdowns caused large spikes in many of our economic indicators that get tracked. One of these indicators that would have been very visible on a daily basis is grocery store sales. It was a prominent story at the beginning of the pandemic with how certain foods or materials were constantly out of stock. Some of this was likely due to production being shut down, but a lot of it was also sales increasing. With these fluctuations, I thought it would be an interesting problem to investigate which of the other main economic indicators track with the sales and if they could be used to predict future sales.

-----------
## Conclusion
Using a polynomial regression model to predict national grocery store sales had mixed results. Before the pandemic, the selected features here did a very good job at predicting grocery store sales. By narrowing the focus around the pandemic however, the features here performed much worse. 
There are a few routes that could be explored from here. 
  1. Explore other features. Look into seeing if different or additional economic indicators improve performance.
  2. There appeared to be a lag between indicators which would require some deeper analysis.  For example, there was a large spike in grocery store sales and unemployment rate last year, but the spike in unemployment rate occurred a month after the increase in grocery store sales.
  3. The large changes in multiple economic indicators as a result of the pandemic might simply be too difficult to model.

-----------
## References
* [Gross Domestic Income](https://fred.stlouisfed.org/series/GDI)
* [Consumer Price Index for Urban Consumers](https://fred.stlouisfed.org/series/CPIAUCSL)
* [Grocery Store Sales](https://fred.stlouisfed.org/series/RSGCS)
* [Unemployment Rate](https://fred.stlouisfed.org/series/UNRATE)
* [Total Consumer Credit](https://fred.stlouisfed.org/series/TOTALSL)
* [US Population](https://fred.stlouisfed.org/series/POPTHM)
* [Real Median Personal Income](https://fred.stlouisfed.org/series/MEPAINUSA672N)
* [10-Year Breakeven Inflation Rate](https://fred.stlouisfed.org/series/T10YIEM)
