'''
01_tensors.py
Introduction to PyTorch Tensors

Tensors are the fundamental data structure in PyTorch.

Acknowledgements: Based on materials from various online resources.
    https://www.coursera.org/learn/pytorch-fundamentals/lecture/pZvZU/tensors

Setup:
    source .venv_pytorch/bin/activate
    pip install -r requirements.txt
    python 01_tensors.py

Authors: Soumya Banerjee
Date: June 2025
'''

import torch
import numpy as np

# teaching resource
# https://www.coursera.org/learn/pytorch-fundamentals/lecture/pZvZU/tensors


my_distances = [ [0.1], [0.31], [0.23], [1.99] ]
tensor_distances = torch.tensor(my_distances,
                                dtype = torch.float32) # create a tensor from a list    

# get shape of tensor
print("\n Shape of Tensor of distances: \n")
print(tensor_distances.shape) # torch.Size([4, 1])

# getting an item from a tensor
print("\n First item in the tensor of distances: \n")
print(tensor_distances[0]) # get the first item
print(tensor_distances[0].item()) # get the first item as a standard python

# now feed it to a model
simple_nn_model = torch.nn.Linear(in_features = 1,
                                  out_features = 1) # simple linear model

# predict
output_nn = simple_nn_model(tensor_distances)
print("Output from simple neural network model:")
print(output_nn)


numpy_array = np.array([12, 89, 6.7, 9.0])
#tensor_from_numpy = torch.from_numpy(numpy_array)


####################################
# tensor math operations
####################################
new_distances = torch.tensor([ [0.19],
                               [0.45],
                               [9.01],
                               [0.89] 
                               ],
                               dtype = torch.float32
                            )

print("\n Tensor Math Operations: \n")
print(new_distances + 1.99) # add 1.99 to each element
print(new_distances * 2.0)  # multiply each element by 2.0

more_distances = torch.tensor( [ [1.9],
                                [9.9],
                                [0.8],
                                [6.7],
                                ],
                                dtype = torch.float32
)

print("\n Element-wise multiplication of two tensors: \n ")
print(new_distances * more_distances) # element-wise multiplication

print("\n dot product of two tensors: \n ")
print( torch.matmul(new_distances.T, more_distances) ) # dot product

######################
# Load from pandas
#######################
import pandas as pd

# load from csv file and convert to tensor
github_url = "https://raw.githubusercontent.com/cambiotraining/ml-unsupervised/main/course_files/data/diabetes_sample_data.csv"
df = pd.read_csv(github_url)
print("\n Dataframe head: \n"
      , df.head()
      )
# convert to tensor
df.values # has all values
tensor_from_pandas = torch.tensor(df.values, 
                                  dtype = torch.float32
                                  )
print("\n Shape of tensor from pandas dataframe: \n")
print(tensor_from_pandas.shape)

print("=" * 50)
print("PYTORCH TENSORS BASICS")
print("=" * 50)

# 1. Creating tensors from scratch
print("\n1. Creating Tensors:")
print("-" * 30)

# Create a 1D tensor (vector)
tensor_1d = torch.tensor([1, 2, 3, 4, 5])
print(f"1D tensor: {tensor_1d}")

# Create a 2D tensor (matrix)
tensor_2d = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"2D tensor:\n{tensor_2d}")

# Create a tensor of zeros
zeros = torch.zeros(2, 3)  # 2 rows, 3 columns
print(f"Zeros tensor:\n{zeros}")

# Create a tensor of ones
ones = torch.ones(2, 3)
print(f"Ones tensor:\n{ones}")

# Create a tensor with random values
random = torch.rand(2, 3)  # Values between 0 and 1
print(f"Random tensor:\n{random}")

# 2. Tensor properties
print("\n2. Tensor Properties:")
print("-" * 30)
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
print(f"Tensor:\n{x}")
print(f"Shape: {x.shape}")  # Dimensions
print(f"Data type: {x.dtype}")  # Type of data
print(f"Device: {x.device}")  # CPU or GPU

# 3. Converting between NumPy and PyTorch
print("\n3. NumPy ↔ PyTorch:")
print("-" * 30)

# NumPy to PyTorch
np_array = np.array([1, 2, 3, 4, 5])
torch_tensor = torch.from_numpy(np_array)
print(f"NumPy array: {np_array}")
print(f"PyTorch tensor: {torch_tensor}")

# PyTorch to NumPy
back_to_numpy = torch_tensor.numpy()
print(f"Back to NumPy: {back_to_numpy}")

# 4. Basic tensor operations
print("\n4. Tensor Operations:")
print("-" * 30)

a = torch.tensor([1, 2, 3])
b = torch.tensor([4, 5, 6])

# Addition
print(f"a + b = {a + b}")

# Multiplication (element-wise)
print(f"a * b = {a * b}")

# Matrix multiplication
matrix_a = torch.tensor([[1, 2], [3, 4]])
matrix_b = torch.tensor([[5, 6], [7, 8]])
result = torch.matmul(matrix_a, matrix_b)
print(f"Matrix multiplication:\n{result}")

# 5. Reshaping tensors
print("\n5. Reshaping:")
print("-" * 30)

original = torch.tensor([1, 2, 3, 4, 5, 6])
print(f"Original: {original}")

reshaped = original.view(2, 3)  # Reshape to 2x3
print(f"Reshaped to 2x3:\n{reshaped}")

reshaped_again = reshaped.view(3, 2)  # Reshape to 3x2
print(f"Reshaped to 3x2:\n{reshaped_again}")

print("\n" + "=" * 50)
print("🎉 Great! You now understand PyTorch tensors!")
print("=" * 50)
