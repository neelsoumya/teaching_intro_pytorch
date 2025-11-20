"""
02_autograd.py
Introduction to Automatic Differentiation (Autograd)

Autograd automatically computes gradients (derivatives).
This is essential for training neural networks!
"""

import torch

print("=" * 50)
print("PYTORCH AUTOGRAD BASICS")
print("=" * 50)

# 1. Basic gradient computation
print("\n1. Computing Gradients:")
print("-" * 30)

# Create a tensor and enable gradient tracking
x = torch.tensor(2.0, requires_grad=True)
print(f"x = {x}")

# Define a function: y = x^2 + 3
y = x**2 + 3
print(f"y = x^2 + 3 = {y}")

# Compute the gradient dy/dx
y.backward()  # This computes the gradient

# The gradient is stored in x.grad
# dy/dx = 2x, so at x=2, gradient = 2*2 = 4
print(f"dy/dx at x=2 is: {x.grad}")

# EXPLANATION
# When you created x with torch.tensor(2.0, requires_grad=True), PyTorch marked x as a leaf tensor that should accumulate gradients. Any operation on tensors with requires_grad=True builds a dynamic computation graph: each output tensor gets a grad_fn (a Function node) that points to the operation that created it.

# What y.backward() does: Calling y.backward() starts backpropagation from y (treating dy/dy = 1 for a scalar). Autograd traverses the graph in reverse, applying the chain rule through each grad_fn, and computes gradients for tensors that require gradients.
# Where the gradient goes: Gradients for leaf tensors are accumulated in their .grad attribute. In this case x.grad receives dy/dx = 2*x, so after y.backward() (with x=2.0) you get x.grad == 4.0.

# Additional notes:

# The graph is dynamic — built as operations run — and freed by default after backward unless you pass retain_graph=True.
# If a tensor is detached or created inside torch.no_grad(), no graph node is created and no gradients are tracked.
# For non-scalar y, you must pass a gradient argument to backward() matching y's shape.


# 2. More complex example
print("\n2. More Complex Function:")
print("-" * 30)

x = torch.tensor(3.0, requires_grad=True)
y = torch.tensor(2.0, requires_grad=True)

# z = x^2 + y^3
z = x**2 + y**3
print(f"x = {x.item()}, y = {y.item()}")
print(f"z = x^2 + y^3 = {z.item()}")

# Compute gradients
z.backward()

# dz/dx = 2x = 2*3 = 6
print(f"dz/dx = {x.grad}")

# dz/dy = 3y^2 = 3*2^2 = 12
print(f"dz/dy = {y.grad}")

# 3. Vector gradients
print("\n3. Vector Operations:")
print("-" * 30)

# Create a vector with gradient tracking
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
print(f"x = {x}")

# Compute y = sum of squares
y = (x**2).sum()  # y = 1^2 + 2^2 + 3^2 = 14
print(f"y = sum(x^2) = {y}")

# Compute gradient
y.backward()

# dy/dx = [2*x[0], 2*x[1], 2*x[2]] = [2, 4, 6]
print(f"dy/dx = {x.grad}")

# 4. Gradient accumulation
print("\n4. Gradient Accumulation:")
print("-" * 30)

x = torch.tensor(5.0, requires_grad=True)

# First computation
y1 = x**2
y1.backward()
print(f"After first backward: x.grad = {x.grad}")

# Gradients accumulate! Need to zero them
x.grad.zero_()  # Reset gradient to zero

# Second computation
y2 = x**3
y2.backward()
print(f"After second backward: x.grad = {x.grad}")

# 5. Detaching from computation graph
print("\n5. Detaching Tensors:")
print("-" * 30)

x = torch.tensor(2.0, requires_grad=True)
y = x**2

# Detach y from the computation graph
y_detached = y.detach()
print(f"y requires grad: {y.requires_grad}")
print(f"y_detached requires grad: {y_detached.requires_grad}")

# 6. Context manager to disable gradients
print("\n6. Disabling Gradients:")
print("-" * 30)

x = torch.tensor(3.0, requires_grad=True)

# With gradients (normal)
y = x**2
print(f"y requires grad: {y.requires_grad}")

# Without gradients (useful for inference)
with torch.no_grad():
    y = x**2
    print(f"Inside torch.no_grad(), y requires grad: {y.requires_grad}")

print("\n" + "=" * 50)
print("🎉 You now understand autograd!")
print("=" * 50)
