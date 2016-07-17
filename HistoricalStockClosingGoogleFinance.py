import sys, urllib.request, json, csv 

import requests, warnings

#import re

from bs4 import BeautifulSoup


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
