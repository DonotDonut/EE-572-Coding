import numpy as np

# Given data
S_load = 1.0 + 0.5j  # Load power in p.u.
Y_line = 1 / (0.02 + 0.06j)  # Admittance of the line in p.u.
Y_cap = 0 + 0.25j  # Admittance of the capacitor in p.u.
V1 = 1.0 + 0j  # Voltage at Bus 1 in p.u.

# Initial guess for V2
V2 = 1.0 + 0j  # Start with a flat start

# Iteration parameters
max_iterations = 100  # Maximum number of iterations

# Line charging admittance split equally on both sides
# Line charging = 5 Mvar 
#Convert % Mar to P.u.  
# P.u. = Mvar / Base MVA 

Y_line_shunt_cap = 0 + 0.05j

# Net admittance matrix at Bus 2
Y2 = Y_line + Y_cap + Y_line_shunt_cap

# Gauss-Seidel iterations
for iteration in range(max_iterations):
    # Previous value of V2 for convergence check
    V2_old = V2

    # Calculate current injection at Bus 2
    I2 = np.conj(S_load / V2)  # Load current

    # Voltage at Bus 2
    V2 = (I2 - Y_line * V1) / - Y2

# Display results
print(f"Voltage at Bus 2: V2 = {V2:.4f} p.u.")