import numpy as np

def jacobi_method(A, b, x0, tol=0.01, max_iter=100):
    """
    Solves a system of linear equations Ax = b using the Jacobi iterative method.

    Parameters:
    A: numpy array, the coefficient matrix
    b: numpy array, the right-hand side vector
    x0: numpy array, the initial guess for the solution
    tol: float, the desired tolerance for the solution
    max_iter: int, the maximum number of iterations

    Returns:
    x: numpy array, the solution vector
    """

    n = len(b)
    x = x0.copy()
    D = np.diag(np.diag(A))
    R = A - D

    for k in range(max_iter):
        x_new = np.zeros(n)
        for i in range(n):
            x_new[i] = (b[i] - np.dot(R[i], x)) / D[i, i]

        if np.linalg.norm(x_new - x) < tol:
            print(f"Converged in {k + 1} iterations.")
            return x_new

        x = x_new

    print("Jacobi method did not converge within the given number of iterations.")
    return x

# Adjust A, b, and x0
A = np.array([[6, 2, 1],
              [4, 10, 2],
              [3, 4, 14]])
b = np.array([3, 4, 2])
x0 = np.array([0, 0, 0])

print("Solving Problem with Jacobi Method")
solution = jacobi_method(A, b, x0)
print("Solution:", solution)
