
import torch
from inkstone.primitives.torch_primitive import EigFunction
# Assuming the EigFunction class is already defined as per your original code

# Test function
def test_eig_backward_function():
    # Create a random square matrix A
    A = torch.randn(3, 3, requires_grad=True)

    # Perform the forward pass through the custom eig function
    e, v = EigFunction.apply(A)

    # Define a simple loss function: sum of eigenvalues
    loss = e.sum().real

    # Perform the backward pass
    loss.backward()

    # Get the computed gradients from the custom eig backward function
    grad_A_custom = A.grad.clone()  # Save the gradients computed by our custom backward function

    # Verify against the built-in PyTorch eig function
    A_clone = A.detach().requires_grad_(True)
    e_builtin, v_builtin = torch.linalg.eig(A_clone)

    # Calculate gradients using PyTorch's built-in eig function
    loss_builtin = e_builtin.sum().real
    loss_builtin.backward()

    # Get the computed gradients from PyTorch's built-in eig backward function
    grad_A_builtin = A_clone.grad.clone()

    # Compare the gradients
    assert torch.allclose(grad_A_custom, grad_A_builtin, atol=1e-6), "Gradients do not match!"

    print("Test passed: Gradients from custom and built-in eig functions match.")

# Run the test
test_eig_backward_function()
