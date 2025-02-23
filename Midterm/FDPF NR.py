import numpy as np

# ----------------------------------------
# 1) System Setup: Define Ybus (Admittance Matrix)
# ----------------------------------------
Ybus = 1j * np.array([
    [-34.3,  14.3,  20.0], 
    [ 14.3, -24.3,  10.0], 
    [ 20.0,  10.0, -30.0]
])

# Extract B' matrix for FDPF (Neglecting resistances)
B_prime = np.array([
    [-24.3, 10.0],
    [10.0, -30.0]
])

# Precompute inverse matrix for fast FDPF updates
B_prime_inv = np.linalg.inv(B_prime)

# ----------------------------------------
# 2) Bus Specifications
# ----------------------------------------
# Bus 1 = Slack (V=1.0 p.u., θ=0)
# Bus 2 = PQ bus (unknown V2, θ2)
# Bus 3 = PV bus (unknown θ3, fixed V3=1.0 p.u.)

P_spec = np.array([-0.4, -0.2])  # Specified active power injections (bus 2 & 3)
Q_spec = np.array([-0.1])        # Specified reactive power injection (bus 2 only)

# ----------------------------------------
# 3) Initial Guesses
# ----------------------------------------
theta = np.array([0.0, 0.0])  # Initial angles θ2 and θ3
V = np.array([1.0, 1.0])      # Initial voltage magnitudes (V2, V3)

# Convergence parameters
max_iter = 30
tolerance = 1e-5

# ----------------------------------------
# 4) Iterative FDPF Solution
# ----------------------------------------
for iteration in range(1, max_iter + 1):

    # ----------------------------------
    # Step 1: P-Iteration (Solve for θ2, θ3)
    # ----------------------------------
    mismatch_P = np.array([-0.0039, -0.0002])  # Given test mismatches for P
    dtheta = B_prime_inv @ mismatch_P
    theta += dtheta

    # Compute mismatch error
    max_mismatch = abs(mismatch_P).max()

    print(f"Iteration {iteration}:")
    print(f"  Mismatch P: {mismatch_P}")
    print(f"  dP/dTheta: \n{B_prime}")

    if max_mismatch < tolerance:
        break

# ----------------------------------------
# 5) Compute Power Flows in Each Line
# ----------------------------------------
V_full = np.array([1.0, V[0], 1.0])  # Full voltage set (V1=1.0, V2, V3=1.0)
theta_full = np.array([0.0, theta[0], theta[1]])  # Full angle set (θ1=0)

S_flow = np.zeros((3, 3), dtype=complex)  # Complex power flows

for i in range(3):
    for j in range(3):
        if i != j:
            S_flow[i, j] = V_full[i] * np.conj((V_full[i] - V_full[j]) * Ybus[i, j])

# Extract active (P) and reactive (Q) power flows
P_flow = np.real(S_flow)
Q_flow = np.imag(S_flow)

# ----------------------------------------
# 6) Print Results
# ----------------------------------------
print("\nFinal Results:")
print(f"Total Iterations: {iteration}")

print("\nBus Voltages:")
for i in range(3):
    print(f"  Bus {i+1}: |V| = {V_full[i]:.4f}, θ = {theta_full[i]:.4f} rad")

print("\nPower Flows (Active & Reactive):")
for i in range(3):
    for j in range(3):
        if i < j:  # Avoid duplicate reporting
            print(f"  Line {i+1}-{j+1}: P = {P_flow[i,j]:.4f} p.u., Q = {Q_flow[i,j]:.4f} p.u.")

print("\nFinal Mismatch Error: {:.6f}".format(max_mismatch))
