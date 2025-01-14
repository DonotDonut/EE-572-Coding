# Let's test the provided code in Python

from pprint import pprint
from numpy import array, zeros, diag, diagflat, dot

def jacobi(A, b, N=11, x=None):
    """Solves the equation Ax=b via the Jacobi iterative method."""
    # Create an initial guess if none is provided
    if x is None:
        x = zeros(len(A[0]))

    # Extract the diagonal elements of A and create the R matrix
    D = diag(A)  # Diagonal elements of A
    R = A - diagflat(D)  # Remaining matrix without the diagonal

    # Perform the iteration for N steps
    for i in range(N):
        x = (b - dot(R, x)) / D
    return x

# Input: Define a 3x3 matrix A, vector b, and an initial guess
A = array([
    [6.0, 2.0, 1.0],
    [4.0, 10.0, 2.0],
    [3.0, 4.0, 14.0]
])
b = array([3.0, 4.0, 2.0])  # Right-hand side vector
guess = array([0.0, 0.0, 0.0])  # Initial guess for x

# Solve the system using the Jacobi method
solution = jacobi(A, b, N=11, x=guess)

# Output the solution
pprint(solution)
