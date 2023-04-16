import numpy as np
import typing

class Layer:
    def __init__(self, l_nodes:int, r_nodes:int):
        self.weights = np.matrix(np.random.rand(r_nodes,l_nodes))*2 - 1
        self.biases = np.matrix(np.random.rand(1,r_nodes))*10
        self.output = None

    def forward(self, A:np.matrix) -> np.matrix:
        self.output = (np.dot(A,self.weights.T) + self.biases)
        return self.output


    def ReLU(self):
        self.output[self.output<0]=0
 
    def softmax(self): 
        e_x = np.exp(self.output - np.max(self.output, axis=1))
        self.output =  e_x / np.sum(e_x, axis=1)

class Network:
    def __init__(self, layers: list[Layer] ):
        self.layers = layers


if __name__ == '__main__':
    np.random.seed(1)

    X = np.matrix([1,2,3,4]).T
    layer1 = Layer(1, 8)
    R1 = layer1.forward(X)
    layer1.ReLU()
    #print( layer1.output )

    layer2 = Layer(8, 8)
    layer2.forward(layer1.output)
    layer2.ReLU()
    print(layer2.output)

    layer3 = Layer(8,1)
    layer3.forward(layer2.output)
    layer3.ReLU()

    print(layer3.output)