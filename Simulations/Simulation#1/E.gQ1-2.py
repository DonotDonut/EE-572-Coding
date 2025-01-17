import numpy as np

# Define the system of linear equations in matrix form: Ax = b
A = np.array([[10, 5],
              [2, 9]])
b = np.array([6, 3])

# Convergence criterion
epsilon = 0.0001  # Tolerance for convergence

# Maximum number of iterations
max_iterations = 1000  # Prevent infinite loops

"""
Solves the system of linear equations Ax = b using the Gauss-Seidel iterative method.

Parameters:
    A (ndarray): Coefficient matrix (2x2).
    b (ndarray): Right-hand side vector.
    epsilon (float): Convergence tolerance.
    max_iterations (int): Maximum number of iterations.

Returns:
    ndarray: Solution vector x.
"""
def gauss_seidel_method(A, b, epsilon, max_iterations):
    # Ensure A is a 2x2 matrix and b is a vector of size 2
    assert A.shape == (2, 2), "Matrix A must be 2x2."
    assert b.shape == (2,), "Vector b must have 2 elements."

    # Initial guess
    x = np.zeros_like(b, dtype=float)
   
    print("Iteration\t x1\t\t x2")
    print("-----------------------------------")
   
    # Iterate
    for iteration in range(1, max_iterations + 1):
        x_new = np.copy(x)

        for i in range(2):
            # Sum over all elements except the diagonal element
            sum_except_i = sum(A[i, j] * x_new[j] for j in range(2) if j != i)
           
            # Update x_new[i]
            x_new[i] = (b[i] - sum_except_i) / A[i, i]

        # Print the values of x1 and x2 for this iteration
        print(f"{iteration}\t\t {x_new[0]:.6f}\t {x_new[1]:.6f}")

        # Check for convergence
        if np.linalg.norm(x_new - x, ord=np.inf) < epsilon:
            print(f"\nConverged in {iteration} iterations.")
            return x_new

        # Update x
        x = x_new

    print("Maximum iterations reached without convergence.")
    return x

# Call main method
if __name__ == "__main__":
    print("Solving Problem with Gauss-Seidel Method")
    solution = gauss_seidel_method(A, b, epsilon, max_iterations)
    print("\nSolution:", solution)
