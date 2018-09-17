# Import dependencies
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from utilities import plot_decision_boundary, load_planar_dataset

# Sigmoid activcation function for output layer
def sigmoid(x):
    return 1/(1+np.exp(-x))
    
# Determine the shape and size of our dataset
def layer_sizes(X, Y):
    n_x = X.shape[0] # size of input layer
    n_h = 4
    n_y = Y.shape[0] # size of output layer
    return (n_x, n_h, n_y)

# Initialize parameters for the entire network
def initialize_parameters(n_x, n_h, n_y):
    
    # Ensure random seed remains consistent
    np.random.seed(2)
    
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Implement forward propagation
def forward_propagation(X, parameters):

    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    
    """
    Z1[i] = W * X + b1
    A1 = tanh(Z1)
    A1 = tanh(Z1)
    ŷ = A2 = σ(Z2)
    yprediction = {1 if A2 > 0.5 0 otherwise
    """
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    
    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache

# Use the Cost Entropy algorthim to compute the cost

def compute_cost(A2, Y, parameters):

    # Number of training examples
    m = Y.shape[1]

    # J = −1/m ∑(Ylog(A2) + (1−Y)log(1−A2)
    cost = np.dot(-1/m, np.sum(np.multiply(Y, np.log(A2)) + np.multiply(1 - Y, np.log(1 - A2))))
    
    # makes sure cost is the dimension we expect.
    cost = np.squeeze(cost)      
                         
    return cost

# Back Propagation
def backward_propagation(parameters, cache, X, Y):
    
    # Size of the training set
    m = X.shape[1]
    # Gather variables from dictionary "parameters"
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation
    """ 
        dZ2 = A2 - Y
        dW2 = 1/m  ∑ dZ2[1]
        dZ1 = W2.T dZ2 * (1 - A1^2)
        dW1 = 1/m ∑ dZ1 X.T
        db1 = 1/m ∑ dZ1[1]
    """
    dZ2 = A2 - Y
    dW2 = 1 / m * np.dot(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1/m * np.dot(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis=1, keepdims=True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

# Update parameters using Gradient Descent
def update_parameters(parameters, grads, learning_rate = 1.2):
  
    # Retrieve parameters from dictionary "parameters" and "grads
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"] 
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    
    # Peform θ = θ − (α * dw/dθ) where θ = a variable and  "dw/dθ" = the darevitive of the variable
    W1 = W1 - np.dot(learning_rate, dW1)
    b1 = b1 - np.dot(learning_rate, db1)
    W2 = W2 - np.dot(learning_rate, dW2)
    b2 = b2 - np.dot(learning_rate, db2)
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters

# Full neural network
def neuralNetwork_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    
    # Set random seed and n_x, n_y variables
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters  
    parameters = initialize_parameters(n_x, n_h, n_y)
    
    # Loop (gradient descent)
    for i in range(0, num_iterations):
         
        # Forward propagation
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function
        cost = compute_cost(A2, Y, parameters)
 
        # Backpropagation
        grads = backward_propagation(parameters, cache, X, Y)
 
        # Gradient descent parameter update.
        parameters = update_parameters(parameters, grads)
        
    return parameters

# Use the output of forward_propagation to form a prediction
def predict(parameters, X):

    A2, cache = forward_propagation(X, parameters)
    predictions = np.where(A2>0.5,1,0)

    return predictions

def main():

    # set a seed so that the results are consistent
    np.random.seed(1) 

    # Load the data from sklearn
    X, Y = load_planar_dataset()

    # Visualize the data:
    plt.scatter(X[0, :], X[1, :], c=Y[0], s=40, cmap=plt.cm.Spectral)
    plt.show()

    # Build a model with a n_h-dimensional hidden layer
    parameters = neuralNetwork_model(X, Y, n_h = 4, num_iterations = 10000, print_cost=True)

    # Plot the decision boundary
    plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
    plt.title("Decision Boundary for hidden layer size " + str(4))
    plt.show() 

    # Print accuracy
    predictions = predict(parameters, X)
    print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')

    # Test the accuracy of the model with a range of different layers
    # This may take a while...

    plt.figure(figsize=(16, 32))
    plt.show()
    hidden_layer_sizes = [1, 2, 3, 4, 5, 20, 50]
    
    for i, n_h in enumerate(hidden_layer_sizes):
        plt.subplot(5, 2, i+1)
        plt.title('Hidden Layer of size %d' % n_h)
        parameters = neuralNetwork_model(X, Y, n_h, num_iterations = 5000)
        plot_decision_boundary(lambda x: predict(parameters, x.T), X, Y)
        predictions = predict(parameters, X)
        accuracy = float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100)
        print ("Accuracy for {} hidden units: {} %".format(n_h, accuracy))


if __name__ == "__main__":
    main()