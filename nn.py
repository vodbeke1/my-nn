import numpy as np 
import scipy as sp 
import math
import random

class Neuron:
    def __init__(self, size_previous, index_):
        
        self.weights = [random.randint(0, 100)/100 for i in range(size_previous)]
        self.length = len(self.weights)
        self.bias = 0
        self.value = None
        self.learning_rate = 0.2
        self.i = index_

    def set_activation(self, _input:list, input_layer=False):
        if input_layer:
            r = 0
            for i in range(self.length):
                r += self.weights[i]*_input[i]
            self.value = r
        else:
            r = 0
            for i in range(self.length):
                r += self.weights[i]*_input[i].value
            self.value = r

    def activation(self, x, bias):
        return (1 / (1 + math.exp(-x))) + bias
    
    def update_weights(self, adjustments):
        for i in range(self.length):
            self.weights[i] += adjustments[i]

    def adjust_weights_output(self, pre_layer, actual):
        self.s = (self.value - actual[self.i])*self.value*(1 - self.value)
        for i in range(self.length):
            self.weights[i] += -1*self.learning_rate*pre_layer[i].value*self.s
    
    def adjust_weights_hidden(self, pre_layer, post_layer):
        diff = 0
        for n in post_layer:
            diff += n.weights[self.i]*n.s

        self.s = diff*self.value*(1 - self.value)
        for i in range(self.length):
            self.weights[i] += -1*self.learning_rate*pre_layer[i].value*self.s

    def adjust_weights_first_hidden(self, input_, post_layer):
        diff = 0
        for n in post_layer:
            diff += n.weights[self.i]*n.s
        
        self.s = diff*self.value*(1 - self.value)
        for i in range(self.length):
            self.weights[i] += -1*self.learning_rate*input_[i]*self.s


class NeuralNet:
    def __init__(self, input_:list, output_:list, internal_layers=[4]):
        self.input_ = input_
        self.output_ = output_
        self.internal_layers = internal_layers
        self.network = []
        
        net_dim = internal_layers
        net_dim.append(len(output_))
        
        for i in range(len(net_dim)):
            if i == 0:
                layer = [Neuron(len(input_), j) for j in range(net_dim[i])]
            else:
                # What is going on here
                layer = [Neuron(net_dim[i-1], j) for j in range(net_dim[i])]
            self.network.append(layer)

    def feed_forward(self):
        for i in range(len(self.network)):
            if i == 0:
                for n in self.network[i]:
                    n.set_activation(self.input_, input_layer=True)
            else:
                for n in self.network[i]:
                    n.set_activation(self.network[i-1])
        

    def back_prop(self):
        index_ = [i for i in range(len(self.network))][::-1]
        for i in index_:
            # Last layer
            if i == index_[0]:
                for n in self.network[i]:
                    n.adjust_weights_output(pre_layer=self.network[i-1], actual=self.output_)
            # First layer
            elif i == index_[-1]:
                for n in self.network[i]:
                    n.adjust_weights_first_hidden(input_=self.input_, post_layer=self.network[i+1])
            # Middle layer
            else:
                for n in self.network[i]:
                    n.adjust_weights_hidden(pre_layer=self.network[i-1], post_layer=self.network[i+1])
    
    def run(self, iterations):
        for _ in range(iterations):
            self.feed_forward()
            self.back_prop()
            print([n.value for n in self.network[-1]])
    
if __name__ == "__main__":
    nn = NeuralNet(input_=[3,4,5,6,7], output_=[0,1], internal_layers=[4, 3])
    nn.run(20)



        