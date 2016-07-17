### Takes as input one files containg the historical closing
### prices of a list of stocks runs data analysis to identify high-level trends 
### the method: find approximate eigenvectors of the Prices covariance matrix 

### Outputs X = 1 new files with a grouping of the stock symbols (eigenvectors) 
### corresponding to the largest X = 1 eigenvalues of the covariance matrix   


import sys

import numpy as np 

import matplotlib.pyplot as plt

import pandas as pd


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


###################################################################################

### TO DO: as an alternative, compute PCA and projections of the Closing Proice 
### data onto the highest eigenspaces 
#[U, S, V] = np.linalg.svd(price_data)

