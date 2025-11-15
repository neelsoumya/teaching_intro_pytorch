'''
Python script to build a neural network using PyTorch for educational purposes.

This script demonstrates the basics of defining a neural network, training it on a simple dataset, and evaluating its performance.

Requirements:
- Python 3.x
- PyTorch
- NumPy

Usage:
1. Install Python 3.x from https://www.python.org/downloads/
2. Install PyTorch by following instructions at https://pytorch.org/get-started/locally/
2. Create a virtual environment (optional but recommended):
   python -m venv venv_pytorch
   source venv_pytorch/bin/activate  # On Windows use `venv_pytorch\Scripts\activate`
   pip install -r requirements.txt

Usage:
    python 03_nn.py

Author: soumya banerjee

'''

import torch # Main PyTorch library
import torch.nn as nn # For neural network modules
import torch.optim as optim # For optimization algorithms
import numpy as np # For numerical operations


# distances for delivery
distances = torch.tensor([  [1.0] , 
                          [2.0] , 
                          [3.0] , 
                          [4.0] , 
                          [5.0] , 
                          [6.0] , 
                          [7.0] 
                          ],
                          dtype = torch.float32
                        )

# delivery times
times = torch.tensor([  [1.5] , 
                       [1.7] , 
                       [3.2] , 
                       [3.8] , 
                       [5.1] , 
                       [5.3] , 
                       [7.2] 
                       ],
                       dtype = torch.float32
                     )

# define the neural network model

model = nn.Sequential(
    nn.Linear(1,1) # One input feature (distance), one output feature (time)
)

# define the loss function and optimizer
loss_function = nn.MSELoss() # Mean Squared Error loss
optimizer = optim.SGD(
    model.parameters(), # Stochastic Gradient Descent optimizer
    lr = 0.01          # Learning rate
)

print("Starting training...\n")

# train the model
num_epochs = 1000
for epoch in range(num_epochs): # Training loop
    optimizer.zero_grad()      # Zero the gradients
    outputs = model(distances) # Forward pass
    loss = loss_function(outputs, times) # Compute loss
    loss.backward()            # Backward pass
    optimizer.step()           # Update weights
    #print("Epoch", epoch + 1, "\n")
    #print("Loss:", loss.item(), "\n")

# plot loss over epochs
import matplotlib.pyplot as plt
#plt.figure()
#plt.plot( range(num_epochs),
#         [loss_function()])

# plot the prediction of the model with the actual data
predicted = model(distances).detach().cpu() # Get predictions
# what is detach() doing here?
# It detaches the tensor from the computation graph, so that no gradients are tracked for it.
# detach() returns a new tensor that shares the same storage but is detached from PyTorch's autograd graph — so operations on it won't be tracked for gradients. 
# Use it before converting to NumPy or lists to avoid autograd errors.

try:
    predicted = model(distances).detach().cpu().numpy() # Get predictions as NumPy array
    distances_plot = distances.cpu().numpy()
    times_plot = times.cpu().numpy() 
except:
    predicted = model(distances).detach().cpu().tolist() # Fallback to list if NumPy conversion fails
    distances_plot = distances.cpu().tolist()
    times_plot = times.cpu().tolist()
    
plt.figure()
plt.plot(distances_plot,
         times_plot,
         'ro',
         label = 'Original data'
         )
plt.plot(distances_plot,
         predicted,
         label = 'Model prediction'
        )
plt.xlabel("Distance")
plt.ylabel("Delivery time")
plt.legend()
plt.show()


# Define a custom model class

class DeliveryTimeModel(nn.Module): # Custom model class
    def __init__(self):
        super(DeliveryTimeModel, self).__init__()
        self.linear = nn.Linear(1, 1)  # One input feature (distance), one output feature (time)

    def forward(self, x):
        return self.linear(x)
    
    def backward(self, x, y):
        # Forward pass
        y_pred = self.forward(x)
        # Compute loss
        loss = loss_function(y_pred, y)
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss.item()

    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
        

    
# instantiate the model
print("\n Creating custom model instance...\n")
model = DeliveryTimeModel()

# train the model
print("\n Starting training of custom model...\n")
num_epochs = 1000
for epoch in range(num_epochs):
    loss = model.backward(distances, times)

print("\n Training completed.\n")
