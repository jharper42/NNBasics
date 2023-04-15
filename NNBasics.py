import numpy as np
import math
import typing

class layer:
    def __init__(self, l_nodes:int, r_nodes:int):
        self.weights = np.matrix(np.random.rand(r_nodes,l_nodes))
        self.biases = np.matrix(np.random.rand(1,r_nodes))

    def forward(self, A:np.matrix) -> np.matrix: 
        return (np.dot(A,self.weights.T) + self.biases)


if __name__ == '__main__':
    np.random.seed(1)

    X = np.matrix([1,0,0,0]).T
    layer1 = layer(1, 3)
    R1 = layer1.forward(X)

    layer2 = layer(3, 2)
    R2 = layer2.forward(R1)

    layer3 = layer(2,1)
    R3 = layer3.forward(R2)
    print(R3)
    