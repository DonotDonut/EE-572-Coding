import numpy as np

# Define the matrix A and vector b for a 2x2 system
A = np.array([[10, 5], [2, 9]])
b = np.array([6, 3])

# Initial guess
x0 = np.array([0, 0])

# Maximum allowed component-wise error percentage
max_error_percentage = 0.01

# Initialize variables
x = x0
n = len(b)
D = np.diag(np.diag(A))
R = A - D
tolerance = 0.0001 

# Display the header for the iteration table
print(f"{'Iteration':<10} {'x1':<15} {'x2':<15}")
print('-' * 50)

# Iteration counter
k = 1

while True:
    # Calculate the new approximation
    x_new = np.linalg.inv(D).dot(b - R.dot(x))
    
    # Calculate the component-wise error
    error = np.abs((x_new - x) / x_new)
    max_error = np.max(error) * 100  # Convert to percentage
    
    # Print the current iteration, the values of x, and the max error percentage
    print(f"{k:<10} {x_new[0]:<15.6f} {x_new[1]:<15.6f}")
    
    # Check if the maximum component-wise error is below the tolerance
    if max_error < max_error_percentage:
        break
    
    # Update the current approximation
    x = x_new
    k += 1

# Summary table
print("\nSummary Table:")
print(f"{'Iteration':<10} {'x1':<15} {'x2':<15}")
print('-' * 50)
print(f"{k:<10} {x_new[0]:<15.6f} {x_new[1]:<15.6f}")
