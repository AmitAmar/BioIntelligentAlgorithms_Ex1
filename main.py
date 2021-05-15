import numpy as np
import pandas as pd
from ann import ANN

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


# activation function
  
def sigmoid(x):
    return(1/(1 + np.exp(-x)))
    
# Creating the Feed forward neural network
# 1 Input layer(1, 30)
# 1 hidden layer (1, 5)
# 1 output layer(3, 3)
  
def f_forward(x, w1, w2):
    # hidden
    z1 = x.dot(w1)# input from layer 1 
    a1 = sigmoid(z1)# out put of layer 2 
      
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    return(a2)
   
# initializing the weights randomly
def generate_wt(x, y):
    l =[]
    for i in range(x * y):
        l.append(np.random.randn())
    return(np.array(l).reshape(x, y))
      
# for loss we will be using mean square error(MSE)
def loss(out, y):
    s =(np.square(out-y))
    s = np.sum(s)/len(y)
    return(s)
    
# Back propagation of error 
def back_prop(x, y, w1, w2, alpha):
      
    # hiden layer
    z1 = x.dot(w1)# input from layer 1 
    a1 = sigmoid(z1)# output of layer 2 
      
    # Output layer
    z2 = a1.dot(w2)# input of out layer
    a2 = sigmoid(z2)# output of out layer
    
    # error in output layer
    d2 =(a2-y)
    d1 = np.multiply((w2.dot((d2.transpose()))).transpose(), 
                                   (np.multiply(a1, 1-a1)))
  
    # Gradient for w1 and w2
    w1_adj = x.transpose().dot(d1)
    w2_adj = a1.transpose().dot(d2)
      
    # Updating parameters
    w1 = w1-(alpha*(w1_adj))
    w2 = w2-(alpha*(w2_adj))
      
    return(w1, w2)
  
def train(x, y, w1, w2, alpha = 0.01, epoch = 10):
    acc =[]
    losss =[]
    for j in range(epoch):
        l =[]
        for i in range(len(x)):
            out = f_forward(x[i], w1, w2)
            l.append((loss(out, y[i])))
            w1, w2 = back_prop(x[i], y[i], w1, w2, alpha)
        print("epochs:", j + 1, "======== acc:", (1-(sum(l)/len(x)))*100)   
        acc.append((1-(sum(l)/len(x)))*100)
        losss.append(sum(l)/len(x))
    return(acc, losss, w1, w2)
   
def predict(x, w1, w2):
    Out = f_forward(x, w1, w2)
    maxm = 0
    k = 0
    for i in range(len(Out[0])):
        if(maxm<Out[0][i]):
            maxm = Out[0][i]
            k = i
    if(k == 0):
        print("Image is of letter A.")
    elif(k == 1):
        print("Image is of letter B.")
    else:
        print("Image is of letter C.")
    plt.imshow(x.reshape(5, 6))
    plt.show()  


def main():
    """
    w1 = generate_wt(30, 5)
    w2 = generate_wt(5, 3)
    print(w1, "\n\n", w2)



    acc, losss, w1, w2 = train(x, y, w1, w2, 0.1, 200)
    """
    x =[np.array(a).reshape(1, 30), np.array(b).reshape(1, 30), 
                                np.array(c).reshape(1, 30)]
    y = np.array(global_y)
    ann = ANN(input_dim=30, output_dim=3, hidden_layers=1, hidden_layer_length=5)
    #print(ann)
    #print(ann.feed_forward(x[0])[-1])

    for i in range(10):
        ann.back_propogation(x[0], y[0], 0.01)

if __name__ == "__main__":
    main()
