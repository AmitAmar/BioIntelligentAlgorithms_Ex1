import numpy as np
from BioIntelligentAlgorithms_Ex1.src.ann import ANN

# Creating data set
  
# A
a =[0, 0, 1, 1, 0, 0,
   0, 1, 0, 0, 1, 0,
   1, 1, 1, 1, 1, 1,
   1, 0, 0, 0, 0, 1,
   1, 0, 0, 0, 0, 1]
# B
b =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 1, 0,
   0, 1, 1, 1, 1, 0]
# C
c =[0, 1, 1, 1, 1, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 0, 0, 0, 0,
   0, 1, 1, 1, 1, 0]
  
# Creating labels
global_y =[[1, 0, 0],
           [0, 1, 0],
           [0, 0, 1]]


def main():
    # Fixes numpy's random seed
    np.random.seed(0)

    # Converts the dataset and the tags to be numpy arrays
    x = [np.array(a).reshape(1, 30),
         np.array(b).reshape(1, 30), 
         np.array(c).reshape(1, 30)]
    y = np.array(global_y)

    # Creates a new ANN to be trained with the data
    ann = ANN(input_dim=30, output_dim=3, hidden_layers=1, hidden_layer_length=5)
    
    # Trains the ANN with the dataset
    ann.train(x, y, alpha=0.1, epochs=200)


if __name__ == "__main__":
    main()
