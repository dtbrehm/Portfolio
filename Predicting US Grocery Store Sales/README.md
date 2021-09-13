# Predicting US Grocery Store Sales

-----------
### Table of Contents

[Introduction](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#introduction)

[Process](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#process)

[Conclusion](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#conclusion)

[References](https://github.com/dtbrehm/Portfolio/tree/main/Predicting%20US%20Grocery%20Store%20Sales#references)

-----------
## Introduction
The recent pandemic has had an effect on many parts of our country, with the economy being one of the clearer ones. The shutdowns caused large spikes in many of our economic indicators that get tracked. One of these indicators that would have been very visible on a daily basis is grocery store sales. It was a prominent story at the beginning of the pandemic with how certain foods or materials were constantly out of stock. Some of this was likely due to production being shut down, but a lot of it was also sales increasing. With these fluctuations, I thought it would be an interesting problem to investigate which of the other main economic indicators track with the sales and if they could be used to predict future sales.

## Process
The data gathered was in the format of individual indicators. Attempting to keep these in the same date format made it simpler to combine. From the sources and structure of the data basically being just the date and one variable column, there wasn’t much cleaning required. With all of the date formats being YYYY-MM-DD, it was very easy to merge as well. After compiling all of the different data sources, features could be eliminated through the process of training the model through Recursive Feature Elimination. Once the model was trained, grocery store sales numbers could be predicted.

The model selected for this analysis was polynomial regression. This is a method of regression analysis where the relationship between the independent variables and dependent variable is modelled with an nth degree polynomial. The features are transformed to fit this selected polynomial degree. For example, if you have three features [a, b, c] and are attempting a polynomial with a degree of two, each row of data then becomes [1, a, b, c, a^2, b^2, c^2, ab, bc, ca].

A concern for this model’s accuracy was the data size. Even if more features are added, the corresponding sales numbers which we were trying to predict were set. As a result of the grocery store sales data being monthly points from 1992, there was only around 350 data points. Through trying to add more features to the model, the data did not align well. There was monthly, quarterly, and annual data. This introduced missing values, not even accounting for the differences in seasonally adjusted vs non-seasonally adjusted data. 

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
