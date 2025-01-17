import numpy as np

def newton_raphson_2x2(f, jacobian, x0, tol=1e-4, max_iter=100):
    """
    Newton-Raphson method for solving a system of two nonlinear equations.

    Parameters:
        f (function): A function that returns the system of equations as a numpy array.
        jacobian (function): A function that returns the Jacobian matrix as a numpy array.
        x0 (numpy array): Initial guess for the solution.
        tol (float): Tolerance for the solution (default: 1e-6).
        max_iter (int): Maximum number of iterations (default: 100).

    Returns:
        numpy array: The solution vector.
    """
    x = x0
    for i in range(max_iter):
        fx = f(x)  # Evaluate the system of equations
        jx = jacobian(x)  # Evaluate the Jacobian matrix
        
        # Solve for delta_x: J(x) * delta_x = -F(x)
        delta_x = np.linalg.solve(jx, -fx)
        
        # Update the solution
        x = x + delta_x

        # Check for convergence
        if np.linalg.norm(delta_x, ord=2) < tol:
            print(f"Converged in {i+1} iterations.")
            return x

    raise ValueError("Newton-Raphson method did not converge within the maximum number of iterations.")

# Example usage
def f(x):
    """
    System of nonlinear equations:
    f1(x, y) = 10 * x1 * sin(x2) + 2
    f2(x, y) = 10 * (x1)^2 - 10 * x1 * cos(x2) + 1
    """
    return np.array([
        10 * x[0] * np.sin(x[1]) + 2,
        10 * (x[0]**2) - 10 * x[0] * np.cos(x[1]) + 1
    ])

def jacobian(x):
    """
    Jacobian matrix of the system of equations:
    J = [[df1/dx1, df1/dx2],
         [df2/dx1, df2/dx2]]
    """
    return np.array([
        [10 * np.sin(x[1]), 10 * x[0] * np.cos(x[1])],
        [20 * x[0] - 10 * np.cos(x[1]), 10 * x[0] * np.sin(x[1])]
    ])

# Initial guess
x0 = np.array([2, 0])

# Solve the system
solution = newton_raphson_2x2(f, jacobian, x0)
print("Solution:", solution)
