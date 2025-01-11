import numpy as np

# Initialized values, Adjust when needed 
# Define the system of linear equations in matrix form: Ax = b
A = np.array([[6, 2, 1],
              [4, 10, 2],
              [3, 4, 14]])
b = np.array([3, 4, 2])

# Convergence criterion
epsilon = 0.01 # tolerance 

# Maximum number of iterations 
max_iterations = 1000 # prevent infinite loops

"""
Solves the system of linear equations Ax = b using the Gauss-Seidel iterative method.

Parameters:
    A (ndarray): Coefficient matrix (3x3).
    b (ndarray): Right-hand side vector.
    epsilon (float): Convergence tolerance 
    max_iterations (int): Maximum number of iterations 

Returns:
    ndarray: Solution vector x.
"""
def gauss_seidel_method(A, b, epsilon, max_iterations):
    # Ensure A is a 3x3 matrix and b is a vector of size 3
    assert A.shape == (3, 3), "Matrix A must be 3x3."
    assert b.shape == (3,), "Vector b must have 3 elements."

    # Initial guess (x0 = [0, 0, 0])
    x = np.zeros_like(b, dtype=float)
    
    # Iterate
    for iteration in range(max_iterations):
        x_new = np.copy(x)

        for i in range(3):
            # Sum over all elements except the diagonal element
            sum_except_i = sum(A[i, j] * x_new[j] for j in range(3) if j != i)
            
            # Update x_new[i]
            x_new[i] = (b[i] - sum_except_i) / A[i, i]

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            print(f"Converged in {iteration + 1} iterations.")
            return x_new

        # Update x
        x = x_new

    print("Maximum iterations reached without convergence.")
    return x

# Call main method
if __name__ == "__main__":
    # Solve using Gauss-Seidel method
    print("Solving Problem with Gauss-Seidel Method")
    solution = gauss_seidel_method(A, b, epsilon, max_iterations)
    print("Solution:", solution)
