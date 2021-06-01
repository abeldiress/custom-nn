import activation_functions as af
import numpy as np

class NeuralNetwork():
    def __init__(self):
        self.synaptic_weights = []
        self.model_structure = None

    ## should be called before all the hidden layers
    def set_input_dim(self, num:int):
        if self.model_structure != None:
            raise Exception('Model Structure declared before number of input neurons')
        else:
            self.model_structure = []
            self.model_structure.append([num, None])
    
    ## should be called after all the hidden layers
    def set_output_dim(self, num:int):
        if self.model_structure == None:
            raise Exception('Number of output neurons declared before any model structure')
        else:
            self.model_structure.append([num, None])
    
    def add_layer(self, num_of_neurons:int, activation_function:str):
        if self.model_structure == None:
            self.model_structure = []
        
        self.model_structure.append([num_of_neurons, activation_function])
    
    def train(self, X_train:np.ndarray, y_train:np.ndarray, epochs:int):
        iterations = epochs * X_train.shape[0]

        if X_train.shape[1] != self.model_structure[0][0]:
            raise Exception('Input layer size and training size don\'t match')
        elif y_train.shape[1] != self.model_structure[-1][0]:
            raise Exception('Output layer size and training size don\'t match')
        else:
            # randomly generates synaptic weights for intialization
            for layer_index in range(len(self.model_structure[1:])):
                selected_portion = self.model_structure[1:]
                layer_weights = []
                for i in range(selected_portion[layer_index][0]):
                    if layer_index == 0:
                        layer_weights.append(2 * np.random.random((self.model_structure[0][0], 1)) - 1)
                    else:
                        layer_weights.append(2 * np.random.random((selected_portion[layer_index-1][0], 1)) - 1)
                
                self.synaptic_weights.append(np.array(layer_weights))
            
            print(self.synaptic_weights)

            for _ in range(iterations):
                pass
                # TODO

X_train = np.array([
    [1,0,0],
    [0,1,0],
    [1,1,0],
    [0,0,1],
])

y_train = np.array([
    [1],
    [0],
    [0],
    [1]
])

nn = NeuralNetwork()
nn.set_input_dim(3)
nn.add_layer(4, 'relu')
nn.add_layer(2, 'relu')
nn.set_output_dim(1)
nn.train(X_train, y_train, 1)

print(nn.model_structure)