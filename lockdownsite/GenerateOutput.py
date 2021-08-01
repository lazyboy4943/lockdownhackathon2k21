import pickle
from math import e

def sigmoid(x):
    return( 1 / (1 + e ** (-1 * x) ) )

Construction = [15, 16, 50, 50, 16, 12]

with open("WeightsAndBiases.txt", "rb") as f:
    WeightsBiases = pickle.load(f)

def Output(image):
    global Neurons
    global TempSums
    Neurons = [[0 for i in range(Construction[n])] for n in range(1, len(Construction))]
    TempSums = [[0 for i in range(Construction[n])] for n in range(1, len(Construction))]
    for n in range(len(Construction) - 1): # Loops through the layers
        for i in range(Construction[n + 1]): # Loops through the output layer
            sum = 0
            for j in range(Construction[n]): # Loops through the input layer
                sum = sum + ( WeightsBiases[n][i][j] * (image[j] if n == 0 else Neurons[n - 1][j]))
            temp = sum + WeightsBiases[n][i][-1]
            Neurons[n][i] = sigmoid(temp)
            TempSums[n][i] = temp

    Largest = 0

    for x in Neurons[-1]:
        if x > Largest:
            Largest = x

    return Neurons[-1].index(Largest)

            
