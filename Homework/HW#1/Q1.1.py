import numpy as np

# Define the matrix A and vector b
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
D = np.diag(np.diag(A))  # Diagonal matrix of A
R = A - D  # R matrix (A - D)
tolerance = max_error_percentage / 100

# Display the header for the iteration table
print('Iteration\t x1\t\t x2\t\t x3\t\t Max Error (%)')
print('-----------------------------------------------------------------------')

# Iteration counter
k = 1

while True:
    # Calculate the new approximation
    x_new = np.linalg.solve(D, b - np.dot(R, x))
    
    # Calculate the component-wise error
    error = np.abs((x_new - x) / x_new)
    max_error = np.max(error) * 100  # Convert to percentage
    
    # Print the current iteration, the values of x, and the max error percentage
    print(f'{k}\t\t {x_new[0]:.6f}\t {x_new[1]:.6f}\t {x_new[2]:.6f}\t {max_error:.2f}%')
    
    # Check if the maximum component-wise error is below the tolerance
    if max_error < max_error_percentage:
        break
    
    # Update the current approximation
    x = x_new
    k += 1

# Summary table
print('\nSummary Table:')
print('Iteration\t x1\t\t x2\t\t x3\t\t Max Error (%)')
print('-----------------------------------------------------------------------')
print(f'{k}\t\t {x_new[0]:.6f}\t {x_new[1]:.6f}\t {x_new[2]:.6f}\t {max_error:.2f}%')
