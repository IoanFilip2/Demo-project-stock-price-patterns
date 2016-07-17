### Takes as input two files containg the historical closing
### prices of a list of stocks and the volumes traded and
### runs data analysis to identify high-level trends 

import sys

import numpy as np 

import matplotlib.pyplot as plt

import networkx as nx


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







###########################################################

## TO DO:

# Incorporating the volumes data from filepath2:

# Applying smoothing techniques to the price data:

# Incorporating Fourier theory to process tick "signal":

# Further data visualization:

