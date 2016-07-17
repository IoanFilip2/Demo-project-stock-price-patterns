import sys

#print(sys.version)

import numpy as np

import matplotlib.pyplot as plt


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