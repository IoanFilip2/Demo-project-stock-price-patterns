import sys, os

import numpy as np 

import pandas as pd

import re


## Takes as input a directory path and produces two csv files

## One file with historical closing prices for past 200 days
## One file with the trading volumes over same period

symbolsList = []

priceMatrix = np.zeros((200, 1))

volumeMatrix = np.zeros((200,1))


path = "C:\Python33\IntroToML\Scraping\HistoricalClosingPricesGoogleFinance"

#testpath = "C:\Python33\IntroToML\Scraping\TestingFolder"

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



