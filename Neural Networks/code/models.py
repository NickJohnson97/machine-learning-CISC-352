import nn

class PerceptronModel(object):
    def __init__(self, dimensions):
        """
        Initialize a new Perceptron instance.

        A perceptron classifies data points as either belonging to a particular
        class (+1) or not (-1). `dimensions` is the dimensionality of the data.
        For example, dimensions=2 would mean that the perceptron must classify
        2D points.
        """
        self.w = nn.Parameter(1, dimensions)

    def get_weights(self):
        """
        Return a Parameter instance with the current weights of the perceptron.
        """
        return self.w

    def run(self, x):
        """
        Calculates the score assigned by the perceptron to a data point x.

        Inputs:
            x: a node with shape (1 x dimensions)
        Returns: a node containing a single number (the score)
        """
        "*** YOUR CODE HERE ***"
        return nn.DotProduct(self.w, x) # Return dot product between weight vector and input

    def get_prediction(self, x):
        """
        Calculates the predicted class for a single data point `x`.

        Returns: 1 or -1
        """
        "*** YOUR CODE HERE ***"
        if nn.as_scalar(self.run(x)) >= 0:  # If dot product is non-negative
            return 1    # Return 1
        else:           # If dot product is negative
            return -1   # Return -1
        

    def train(self, dataset):
        """
        Train the perceptron until convergence.
        """
        "*** YOUR CODE HERE ***"
        while True:
            accuracy = True # 100% accuracy

            batch_size = 1
            for x,y in dataset.iterate_once(batch_size):    # Iterate over dataset
                pred = nn.as_scalar(y)                      # Prediction of point y
                if pred != self.get_prediction(x):          # If prediction not converged
                    accuracy = False                        # Not 100% accurate
                    nn.Parameter.update(self.w, x, pred)    # Update changed
                    
            if accuracy == True: # If 100% accuracy
                break           # Break loop

class RegressionModel(object):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
    
        h = 30
        
        self.W1 = nn.Parameter(1, h)
        self.b1 = nn.Parameter(1, h)
        
        self.W2 = nn.Parameter(h, 1)
        self.b2 = nn.Parameter(1, 1)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
        Returns:
            A node with shape (batch_size x 1) containing predicted y-values
        """
        "*** YOUR CODE HERE ***"
        # Apply first layer, add first bias (b1) to product of x and first weight (W1 · x)
        #       Method:
        #       linear matrix multiplication of x and W1 using nn.Linear(x, W1) 
        #       add b1 to miltiplied features using nn.AddBias(linear_mult_product, b1)
        layer_1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        # Apply relu function to layer 1
        layer_1 = nn.ReLU(layer_1)
        # Apply second layer, add bias (b2) to product of first layer and second weight (layer_1 · W2)
        layer_2 = nn.AddBias(nn.Linear(layer_1, self.W2), self.b2)
        return layer_2  # Return layer_2
        

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        Inputs:
            x: a node with shape (batch_size x 1)
            y: a node with shape (batch_size x 1), containing the true y-values
                to be used for training
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SquareLoss(self.run(x),y) # Return loss between x (current) and y (true) values

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x,y in dataset.iterate_once(self.batch_size):   # Iterate over dataset
                loss = self.get_loss(x, y)                      # Get loss
                gradient = nn.gradients(loss, [self.W1, self.W2, self.b1, self.b2]) # Get gradient for weights/bias

                # Update each weight/bias using the gradient
                self.W1.update(gradient[0], -0.005) 
                self.W2.update(gradient[1], -0.005)
                self.b1.update(gradient[2], -0.005)
                self.b2.update(gradient[3], -0.005)

            # Stop training when loss is 0.02 or better
            if nn.as_scalar(self.get_loss(nn.Constant(dataset.x), nn.Constant(dataset.y))) <= 0.02: 
                break

class DigitClassificationModel(object):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        # Initialize your model parameters here
        "*** YOUR CODE HERE ***"
        self.batch_size = 1
    
        h = 78
        
        self.W1 = nn.Parameter(784, h)
        self.b1 = nn.Parameter(1, h)
        
        self.W2 = nn.Parameter(h, 10)
        self.b2 = nn.Parameter(1, 10)

    def run(self, x):
        """
        Runs the model for a batch of examples.

        Your model should predict a node with shape (batch_size x 10),
        containing scores. Higher scores correspond to greater probability of
        the image belonging to a particular class.

        Inputs:
            x: a node with shape (batch_size x 784)
        Output:
            A node with shape (batch_size x 10) containing predicted scores
                (also called logits)
        """
        "*** YOUR CODE HERE ***"
        # Apply first layer, add first bias (b1) to product of x and first weight (W1 · x)
        #       Method:
        #       linear matrix multiplication of x and W1 using nn.Linear(x, W1) 
        #       add b1 to miltiplied features using nn.AddBias(linear_mult_product, b1)
        layer_1 = nn.AddBias(nn.Linear(x, self.W1), self.b1)
        # Apply relu function to layer 1
        layer_1 = nn.ReLU(layer_1)
        # Apply second layer, add bias (b2) to product of first layer and second weight (layer_1 · W2)
        layer_2 = nn.AddBias(nn.Linear(layer_1, self.W2), self.b2)
        return layer_2  # Return layer_2

    def get_loss(self, x, y):
        """
        Computes the loss for a batch of examples.

        The correct labels `y` are represented as a node with shape
        (batch_size x 10). Each row is a one-hot vector encoding the correct
        digit class (0-9).

        Inputs:
            x: a node with shape (batch_size x 784)
            y: a node with shape (batch_size x 10)
        Returns: a loss node
        """
        "*** YOUR CODE HERE ***"
        return nn.SoftmaxLoss(self.run(x), y)   # Return loss between x (current) and y (true) values

    def train(self, dataset):
        """
        Trains the model.
        """
        "*** YOUR CODE HERE ***"
        while True:
            for x,y in dataset.iterate_once(self.batch_size):   # Iterate over dataset
                loss = self.get_loss(x, y)                      # Get loss
                gradient = nn.gradients(loss, [self.W1, self.W2, self.b1, self.b2]) # Get gradient for weights/bias

                # Update each weight/bias using the gradient
                self.W1.update(gradient[0], -0.004) 
                self.W2.update(gradient[1], -0.004)
                self.b1.update(gradient[2], -0.004)
                self.b2.update(gradient[3], -0.004)

            # Stop training when loss is 0.02 or better
            if dataset.get_validation_accuracy() >= 0.975: 
                break

