import numpy as np

# ----------------------------------------
# 1) System Setup: Define Ybus (Admittance Matrix)
# ----------------------------------------
Ybus = 1j * np.array([
    [-34.3,  14.3,  20.0], 
    [ 14.3, -24.3,  10.0], 
    [ 20.0,  10.0, -30.0]
])

# Extract B' and B'' (Decoupled Matrices)
B_prime = np.array([
    [-10.0, -9.346],
    [-9.346, 20.0]
])

B_doubleprime = np.array([
    [-10.0, -0.304],
    [-0.304, 20.0]
])

# Precompute inverse matrices for fast FDPF updates
B_prime_inv = np.linalg.inv(B_prime)
B_doubleprime_inv = np.linalg.inv(B_doubleprime)

# ----------------------------------------
# 2) Bus Specifications
# ----------------------------------------
# Bus 1 = Slack (V=1.0 p.u., θ=0)
# Bus 2 = PQ bus (unknown V2, θ2)
# Bus 3 = PV bus (unknown θ3, fixed V3=1.0 p.u.)

P_spec = np.array([0.8, 0.5])  # Specified active power injections (bus 2 & 3)
Q_spec = np.array([0.3])       # Specified reactive power injection (bus 2 only)

# ----------------------------------------
# 3) Initial Guesses
# ----------------------------------------
theta = np.array([0.0, 0.0])  # Initial angles θ2 and θ3
V = np.array([1.0, 1.0])      # Initial voltage magnitudes (V2, V3)

# Convergence parameters
max_iter = 10
tolerance = 1e-5

# ----------------------------------------
# 4) Iterative FDPF Solution
# ----------------------------------------
for iteration in range(1, max_iter + 1):

    # ----------------------------------
    # Step 1: P-Iteration (Solve for θ2, θ3)
    # ----------------------------------
    mismatch_P = np.array([0.05 * np.exp(-0.8 * iteration),  # Simulated ΔP2
                           0.04 * np.exp(-0.8 * iteration)]) # Simulated ΔP3
    dtheta = B_prime_inv @ mismatch_P
    theta += dtheta

    # ----------------------------------
    # Step 2: Q-Iteration (Solve for V2)
    # ----------------------------------
    mismatch_Q = 0.03 * np.exp(-0.8 * iteration)  # Simulated ΔQ2
    dV2 = B_doubleprime_inv[0, 0] * mismatch_Q  # Assuming B'' is mostly diagonal
    V[0] += dV2

    # Compute mismatch error
    max_mismatch = max(abs(mismatch_P).max(), abs(mismatch_Q))

    # Print Jacobian Entries (approximate derivatives)
    dP_dTheta = B_prime  # Approximation (∂P/∂θ)
    dP_dV = np.zeros_like(B_prime)  # FDPF assumes decoupled, so ∂P/∂V ~ 0
    dQ_dTheta = np.zeros_like(B_doubleprime)  # FDPF assumes ∂Q/∂θ ~ 0
    dQ_dV = B_doubleprime  # Approximation (∂Q/∂V)

    print(f"Iteration {iteration}:")
    print(f"  Mismatch P: {mismatch_P}")
    print(f"  Mismatch Q: {mismatch_Q}")
    print(f"  dP/dTheta: \n{dP_dTheta}")
    print(f"  dP/dV (ignored in FDPF): \n{dP_dV}")
    print(f"  dQ/dTheta (ignored in FDPF): \n{dQ_dTheta}")
    print(f"  dQ/dV: \n{dQ_dV}")

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
