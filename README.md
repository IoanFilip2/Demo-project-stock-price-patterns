# Project description 

At a high level, our aim in this project is threefold:
1. to tabulate high-resolution historical tick data going back to the IPO for all the symbols listed on major exchanges like NASDAQ
2. to identify clusters of stock symbols whose prices are highly dependent on one another, as measured by statistical methods applied to their price histories
3. to define a rigorous notion of "market shock" and then construct models to estimate the future effects of market shocks on the stock prices and on the behavior of stock symbol clusters 

Note: In order to define "market shocks" in goal No. 3, we shall need to enhance the stock price tabulation possibly with other quantifiable socio-economic indicators as well.


## Motivation

Understanding and taking advantage through arbitrage opportunities of the stock prices of major companies on the main exchanges is a long-standing problem of general interest. We are interested here in a particular aspect of this major undertaking: using standard statistical tools to cluster stocks not only in twos (as in pairs trading) but in statistically significant cliques according to historical data.


## Demo description

At the current stage of the project, we present a rough draft towards the goals listed in the project description above.

So far, our demo only addresses goal Nos. 1 and 2 from the project description. We deal with a relatively small, low-resolution tick data set and simplistic modeling using multi-linear regression.

All csv files we obtain from scraping and processing, along with several extra plots and digrams can be found here. The "PARTS" below refer to the sections in the file "StockDataPricesALLinONE.py".

### Code summary:

1. We scrape the Google finance website for the closing prices ONLY of 1889 stock symbols from NASDAQ (those for which we found the information). We only consider the period 09/24/2015 - 07/11/2016 (200 days): this is PART 1 of the script.

2. We then tabulate all the stock prices into a single csv file containing the historical price data: PART 2 of the script.

3. PARTS 3 and 4 of the script deal with the data processing: First, we create the covariance matrix for the stock symbols using the 200 tick data points for each to calculate the covariances. This covariance matrix can be interpreted as the adjacency matrix of the weigthed graph of stock symbols, weighted by historical correlation (over the 200 day period). A close-up view of a part of this graph is PLOT 1. Then, we single out an approximate eigenvector for the largest eigenvalue of the covariance matrix of stock prices as follows: we take the subset of the largest entries (in absolute value) of the eigenvector for the largest eigenvalue to single out the corresponding subset of stock symbols. Indeed, the set of stock symbols selected in this manner, 44 out of the total 1889 graph nodes, is the highest connectivity subgraph of our correlation graph.

4. PART 5 runs a linear regression algorithm on the clique of stock symbols obtained from PART 4 of the script. This is a multi-linear regression implemented with gradient descent. The learning curve of this gradient descent procedure is PLOT 2. The output is contained in the file "LinearRegressionOnHighestStockClique.txt" on the github repo; it contains the coefficients of the linear combination. It is given as the last symbol of the cluster list expressed as a linear combination of the other 43 symbols.

Subsequently, a multi-group trading algorithm can be implemented using this linear combination to take advantage of the historical correlation between the 44 stock symbols.


### Future work: 

1. We need better resolution of the tick data over longer periods (minute prices - or better - and multi-year records). All this enormous data is public, but not very easy to find (and scrape) due to the restrictive terms of use of the various online services who provide the data (e.g. Bloomberg finance has made this tick data public, but has explictly disallowed scraping and bots).

2. We need to incorporate stochastic models to account for some of the random noise in the historical price data.

3. We need to further incorporate the historical trading volumes (data we already scraped from Google Finance) into our calculations.

4. We need to run more sophisticated clustering algorithms on the correlation graph of stock symbols in order to isolate smaller highly connected sets of stocks.

5. We need to begin addressing goal No. 3 initially stated in the Project description defining and modeling "market schocks".
 





