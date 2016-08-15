##### IOAN FILIP

##### Stock prices: patterns -- ALL scripts into one file


import sys, os, urllib.request, json, csv 

import requests, warnings

import re

from bs4 import BeautifulSoup

import pandas as pd

import matplotlib.pyplot as plt

import networkx as nx


##############################################################################

# PART 1: Getting the Stock Prices, writing to separate Files

# Input filename containg list of Stock symbols
filename = open(sys.argv[1], 'r')

symbolsList = []

for line in filename:
	symbolsList.append(line.replace('\n', ''))

filename.close()

#symbolsList = ['AAPL']

def historyUrl(symbol):
	return "https://www.google.com/finance/historical?q=NASDAQ%3A"+ symbol +"&ei=-PqCV4jQHMKtmAHOpKuwBg&start=0&num=250"

for symbol in symbolsList:

	url = historyUrl(symbol)

	try:
		# Will raise warning due to Insecure verification
		with warnings.catch_warnings():
			warnings.simplefilter('ignore')
			response = requests.get(url, verify = False)
	except:
		continue

	html = response.content

	soup = BeautifulSoup(html, "html5lib")

	# Imperfect data, exception raised below if table not found
	try:
		tableWrapper = soup.find('div', attrs = {'id': 'prices'})
		infoTable = tableWrapper.find('table', attrs = {'class': 'gf-table historical_price'})
	except:
		continue

	# Extracting the desired information (date, closing price, volume):
	allRows = []
	for row in infoTable.find_all('tr'): 
		newRow = []

		selector = 0
		for cell in row.findAll('td'):
			if selector == 0:
				newRow.append(cell.text.strip())
			if selector in [4, 5]:
				number = ''.join(cell.text.strip().split(','))
				newRow.append(number)

			selector = (selector + 1)%6

		allRows.append(newRow)
		
	#print(allRows)

	outputFilePath = "./HistoricalClosingPricesGoogleFinance/HistoricalData"+symbol+".csv"

	csvFile = open(outputFilePath, 'w+', newline = '')
	infoWriter = csv.writer(csvFile, delimiter = ',')
	#print("This is full table:")
	#print(allRows)
	infoWriter.writerows(allRows[1:])
	
	csvFile.close()


##############################################################################

# PART 2: Tabulating the price data into single csv file

## Takes as input a directory path and produces two csv files

## One file with historical closing prices for past 200 days
## One file with the trading volumes over same period

symbolsList = []

priceMatrix = np.zeros((200, 1))

volumeMatrix = np.zeros((200,1))

## PUT CORRECT PATH
path = ".\HistoricalClosingPricesGoogleFinance"


docs = os.listdir(path)

regex = 'HistoricalData(.*).csv'
pattern = re.compile(regex)

for doc in docs:
	stockSymbol = re.findall(pattern, doc)[0]

	# Imperfect data, not enough information collected for a full 2 columns
	try:
		data_matrix = np.genfromtxt(path+"\\"+doc, delimiter = ',')[:, 2:4]
	except:
		continue

	[rows, cols] = np.shape(data_matrix)

	closing_prices = data_matrix[:, 0]
	closing_prices = closing_prices.reshape(rows, 1)
	daily_volumes = data_matrix[:, 1]
	daily_volumes = daily_volumes.reshape(rows, 1)

	# Padding the arrays with NaN entries to have equal sizes
	if rows != 200:
		pad_size = 200 - rows

		pad = np.empty((pad_size, 1))
		pad[:] = np.NaN

		closing_prices = np.row_stack((closing_prices, pad))
		daily_volumes = np.row_stack((daily_volumes, pad))

	priceMatrix = np.column_stack((priceMatrix, closing_prices))
	volumeMatrix = np.column_stack((volumeMatrix, daily_volumes))

	symbolsList.append(stockSymbol)


priceMatrix = priceMatrix[:, 1:]
volumeMatrix = volumeMatrix[:, 1:]


# Output data frames into separate files:
priceData = pd.DataFrame(priceMatrix, columns = symbolsList)
priceData.to_csv('CompiledHistoricalPrices.csv', header = True, sep = ',')

volumeData = pd.DataFrame(volumeMatrix, columns = symbolsList)
volumeData.to_csv('CompiledHistoricalVolumes.csv', header = True, sep = ',')



##############################################################################

# PART 3: Data processing, creating the covariance matrix (= adjancecy matrix of price correlation graph)
# A closeup of this graph is the submitted PLOT 1

### Takes as input two files containg the historical closing
### prices of a list of stocks and the volumes traded and
### runs data analysis to identify high-level trends 


## Inputting the data:

# Expect: Historical price data
filepath1 = sys.argv[1]

# Expect: Historical trading volumes
#filepath2 = sys.argv[2]


f = open(filepath1, 'r')

# The first row of the Prices Data Frame contains tick symbols
symbolsList = f.readline().split(',')
# The first entry is empty in the Data Frame
symbolsList = symbolsList[1:]
# Removing the extra '\n' symbol from the last stocj symbol
symbolsList[-1] = symbolsList[-1].strip()

f.close()

price_data = np.genfromtxt(filepath1, delimiter = ',')[1:, 1:]
#volume_data = np.genfromtxt(filepath2, delimiter = ',')[1:, 1:]


### BASIC DATA ANALYSIS & VISUALIZATION: 

# Computing Correlation matrix dealing with NaN entries:

[num_rows, num_columns] = np.shape(price_data)

for columnIndex in range(num_columns):
	column = price_data[:, columnIndex]
	column = column.reshape(num_rows, 1)

	num_nonNaN_values = np.count_nonzero(~np.isnan(column))

	# Normalizing each column:
	mean = np.nansum(column)/num_nonNaN_values	

	stdev = np.sqrt( np.nansum( column*column )/num_nonNaN_values - mean*mean )

	column = (column - mean)/stdev

	column = column.reshape(1, num_rows)

	price_data[:, columnIndex] = column


# Replacing NaN values by zeros for the purpose of 
# computing the correlation matrix of the price data
price_data = np.nan_to_num(price_data)


# Computing the covariance matrix of all prices:
# Normalizing by the number of columns penalizes the stocks with shorter closing price history
cov_priceData = (price_data.transpose()).dot(price_data)/num_rows


#Sanity checks:
#print(np.shape(cov_priceData))
#print(np.max(cov_priceData))



## Data visualization:

## Creating a graph with stock symbols as nodes and 
## correlation values as edge-weights

G = nx.Graph()

for rowIndex in range(num_columns):
	for columnIndex in range(rowIndex + 1):
		G.add_edge(symbolsList[rowIndex], symbolsList[columnIndex], weight = cov_priceData[rowIndex, columnIndex])


edge_array = [ d['weight'] for (x, y, d) in G.edges(data=True)]
print(len(edge_array))
plt.hist(edge_array)
plt.savefig(".\DiagramsAndVisualization\covHistogram.jpeg") 
plt.show()



## Arbitrarily fix a large correlation value at 0.9 
larger_edges = [(x,y) for (x,y,d) in G.edges(data=True) if d['weight'] > 0.9]
#smaller_edges = [(x,y) for (x,y,d) in G.edges(data=True) if d['weight'] <= 0.5]

## Drawing the graph using a force-directed layout from the Networkx package
## by only representing the edges with high correlation value
pos = nx.spring_layout(G)

nx.draw_networkx_nodes(G, pos, node_size = 20)
nx.draw_networkx_edges(G, pos, edgelist = larger_edges, width = 2, alpha = 0.2, edge_color = 'b')
#nx.draw_networkx_edges(G, pos, edgelist = smaller_edges, width = 2, alpha = 0.2, edge_color = 'b', style = 'dashed')
nx.draw_networkx_labels(G,pos,font_size = 10, font_family='sans-serif')

plt.axis('off')
plt.savefig(".\DiagramsAndVisualization\covHistoricalPrices_graph.jpeg") 
plt.show()


## TO DO:

# Incorporating the volumes data from filepath2:

# Applying smoothing techniques to the price data:

# Incorporating Fourier theory to process tick "signal":

# Further data visualization:



##############################################################################

# PART 4: Data processing, selects a subset of stock symbols corresponding to the highest connectivity
# clique of stock symbols 

### Takes as input one files containg the historical closing
### prices of a list of stocks runs data analysis to identify high-level trends 
### the method: find approximate eigenvectors of the Prices covariance matrix 

### Outputs X = 1 new files with a grouping of the stock symbols (eigenvectors) 
### corresponding to the largest X = 1 eigenvalues of the covariance matrix   


filepath1 = sys.argv[1]

f = open(filepath1, 'r')

# The first row of the Prices Data Frame contains tick symbols
symbolsList = f.readline().split(',')
# The first entry is empty in the Data Frame
symbolsList = symbolsList[1:]
# Removing the extra '\n' symbol from the last stocj symbol
symbolsList[-1] = symbolsList[-1].strip()

f.close()


price_data = np.genfromtxt(filepath1, delimiter = ',')[1:, 1:]
#volume_data = np.genfromtxt(filepath2, delimiter = ',')[1:, 1:]



### BASIC DATA ANALYSIS & VISUALIZATION: 

# Computing Correlation matrix dealing with NaN entries:

[num_rows, num_columns] = np.shape(price_data)

for columnIndex in range(num_columns):
	column = price_data[:, columnIndex]
	column = column.reshape(num_rows, 1)

	num_nonNaN_values = np.count_nonzero(~np.isnan(column))

	# Normalizing each column:
	mean = np.nansum(column)/num_nonNaN_values	

	stdev = np.sqrt( np.nansum( column*column )/num_nonNaN_values - mean*mean )

	column = (column - mean)/stdev

	column = column.reshape(1, num_rows)

	price_data[:, columnIndex] = column


# Replacing NaN values by zeros for the purpose of 
# computing the correlation matrix of the price data
price_data = np.nan_to_num(price_data)


# Computing the covariance matrix of all prices:
# Normalizing by the number of columns penalizes the stocks with shorter closing price history
cov_priceData = (price_data.transpose()).dot(price_data)/num_rows


## Finding approximate eigenvectors:
[eigen_val, eigen_vect] = np.linalg.eig(cov_priceData)

largest_eigenVal = eigen_val[0]
largest_eigenVect = eigen_vect[:,0]
print("The norm of the eigenvector for the largest eigenvalue:")
print(np.sum(largest_eigenVect*largest_eigenVect))


max_entry = np.max(eigen_vect[:, 0])

## An arbitrary margin for eigenvector coordiate to be counted among highest clique
relevance_threshold = 0.999

approx_eigenVect = np.vectorize(lambda x: int( abs(x) >= relevance_threshold*max_entry ))(largest_eigenVect)

print("The number of relevant stocks in largest eigenvalue grouping:")
print(np.sum(approx_eigenVect))

## Selecting the appropriate stock symbols and building the (column-normalized) price data matrix:
symbolSelection = []
bool_indices = approx_eigenVect > 0

for index in range(num_columns):
	if bool_indices[index]:
		symbolSelection.append(symbolsList[index])

selected_priceData = price_data[:, bool_indices]

approx_eigenVect = largest_eigenVect * approx_eigenVect

## Calculating the error for the approximate eigenvector for the largest eigenvalue
difference_vect = cov_priceData.dot(approx_eigenVect) - eigen_val[0]*approx_eigenVect

error = np.sum(difference_vect*difference_vect)
print("The norm of the matrix difference for the approximate eigenvector is:")
print(error)


## Output data frame containing most relevant grouping of stocks from largest eigenvalue:
selected_priceDataFrame = pd.DataFrame(selected_priceData, columns = symbolSelection)
selected_priceDataFrame.to_csv('StockSelectionHistoricalPrices.csv', header = True, sep = ',')



### TO DO: as an alternative, compute PCA and projections of the Closing Proice 
### data onto the highest eigenspaces 
#[U, S, V] = np.linalg.svd(price_data)




##############################################################################

# PART 5: Runs multi-linear regression by gradient descent on the chosen subset of highest-correlated stock symbols
# The learning curves for gradient descent with various step sizes applied to this data constitute PLOT 2
# The output is a linear combination of the highest connected stocks (44 out of the total 1889)


## Output file:
out = open('.\DiagramsAndVisualization\LinearRegressionOnHighestStockClique.txt', 'w+')

## Read data from table

# Assumes filename given as first input to the script:
filename = sys.argv[1]

my_data = np.genfromtxt(filename, delimiter = ',')

[r, c] = np.shape(my_data)

X = my_data[:, 0:c-1]
X.reshape(r, c-1)

y = my_data[:, c-1]
y.reshape(r, 1)

# Normalizing features (i.e. columns):
def normalize(data):
    [r, c] = np.shape(data)

    means = np.mean(data, axis = 0)
    stdevs = np.std(data, axis = 0)

    means = np.ones((r, c))*means
    stdevs = np.ones((r,c))*stdevs

    data = (data - means)/stdevs
    return data


#### Functions for running gradient descent (vectorised)

def computeCost(X, y, theta):
    m = np.size(y)

    difference = X.dot(theta) - y

    norm = (difference.transpose()).dot(difference)

    return (1/2*m)*norm[0,0]


## Gradient descent implementation


def gradientDescent(X, y, theta, alpha, iterations):
    m = np.size(y)
    costHistory = np.zeros((1, iterations))
    y = y.reshape(m,1)
    theta.reshape(c, 1)

    for iteration in range(iterations):
        
        interMultiply = (X.dot(theta)).reshape(m,1)
        #print(interMultiply)
        theta -= alpha/m * (X.transpose()).dot(interMultiply - y)
        #print(theta)
        costHistory[0, iteration] = computeCost(X, y, theta)

    return [theta, costHistory]


# Initializing the gradient descent:
theta = np.zeros((c, 1))
alpha = 0.01
iterations = 500

# Normalizing the input data, if input already normalized, skip:
#X = normalize(X)
#print(X)

# Reshaping data with column of ones:
X = np.column_stack((np.ones((r, 1)), X))
X = X.reshape(r, c)

#Running gradient descent:
[theta, costHistory] = gradientDescent(X, y, theta, alpha, iterations)
#print(np.shape(costHistory))

## Plotting the convergence graph,
## and saving it to file:
fig  = plt.figure()
plt.plot(list(range(iterations)), costHistory[0, :], 'b-')
plt.title('Gradient cost function convergence graph')
plt.xlabel('Number of iterations')
plt.ylabel('Cost function')

plt.savefig(".\DiagramsAndVisualization\ConvergenceGraphForLinearRegressionOnHighestCliqueStocks.jpeg") 
plt.show()


# Calculating optimum with pseudo-inverse calculation:
theta2 = (np.linalg.inv((X.transpose()).dot(X))).dot(X.transpose()).dot(y)

# Printing minimum and comparison
out.write("The minimum obtained by gradient descent is:")
out.write(np.array_str(theta))
out.write('\n')
out.write("The minimum obtained by pseudo-inverse calculation:")
out.write(np.array_str(theta2))

# Selecting learning rates
#out.write("Selecting learning rates by trial and error:")

thetaNew = np.zeros((c, 1))
numIterations = 100
possibleValues = [0.003, 0.01, 0.03]
# extra possibleValue: 0.1, 0.3
colors = {0.003: 'b', 0.01: 'k', 0.03: 'r'} 
# other colors: , 0.3: 'g', 0.1: 'y'


fig2 = plt.figure()

for val in possibleValues:
    [tempTheta, costHistory] = gradientDescent(X, y, thetaNew, val, numIterations)
    plt.plot(list(range(numIterations)), costHistory[0, :], colors[val]+'-', label="Rate = "+str(val))
    

plt.title('Gradient cost function convergence graphs')
plt.xlabel('Number of iterations')
plt.ylabel('Cost functions with various learning rates')
plt.legend(loc=1)

plt.savefig(".\DiagramsAndVisualization\ChoosingLearningRatesForLinearRegressionOnHighestCliqueStocks.jpeg") 
plt.show()

# Closing file:
out.close()
