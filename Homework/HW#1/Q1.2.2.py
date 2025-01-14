import numpy as np

# Initialized values, Adjust when needed
# Define the system of linear equations in matrix form: Ax = b
A = np.array([[6, 2, 1],
              [4, 10, 2],
              [3, 4, 14]])
b = np.array([3, 4, 2])

# Initial guess
x0 = np.array([0.0, 0.0, 0.0])

# Maximum allowed component-wise error percentage
max_error_percentage = 5

# Initialize variables
x = x0
n = len(b)
tolerance = 0.01 
print("Solving Problem with Gauss-Seidel Iterative Method")
print("")

# Display the header for the iteration table
print('Iteration\t x1\t\t x2\t\t x3\t\t Max Error (%)')
print('-----------------------------------------------------------------------')

# Iteration counter
k = 1

while True:
    x_new = np.copy(x)  # Copy the current x to use for updating

    # Update each component of x sequentially
    for i in range(n):
        # Calculate the sum of A[i, j] * x[j] for j ≠ i
        summation = np.dot(A[i, :], x_new) - A[i, i] * x_new[i]
        x_new[i] = (b[i] - summation) / A[i, i]
    
    # Calculate the component-wise error
    error = np.abs((x_new - x) / x_new)
    max_error = np.max(error) * 100  # Convert to percentage
    
    # Print the current iteration, the values of x, and the max error percentage
    print(f'{k}\t\t {x_new[0]:.6f}\t {x_new[1]:.6f}\t {x_new[2]:.6f}\t {max_error:.2f}%')
    
    # Check if the maximum component-wise error is below the tolerance
    if max_error < max_error_percentage:
        break
    
    # Update the current approximation
    x = np.copy(x_new)
    k += 1

# Summary table
print('\nSummary Table:')
print('Iteration\t x1\t\t x2\t\t x3\t\t Max Error (%)')
print('-----------------------------------------------------------------------')
print(f'{k}\t\t {x_new[0]:.6f}\t {x_new[1]:.6f}\t {x_new[2]:.6f}\t {max_error:.2f}%')
