import numpy as np

###############################################################################
# 1) GIVEN SYSTEM DATA
###############################################################################

# Complex unit for convenience
j = 1j

# -------------------------------------------------------------------
# Bus data:
#   Bus 1: Slack  => V1 = 1.0 + 0j
#   Bus 2: PV     => PG2 = 0.6661 p.u.,  V2 = 1.05 p.u.,  Q2 unknown
#   Bus 3: PQ     => S3 = 2.8653 + j1.2244  (Load)
# -------------------------------------------------------------------
V1 = 1.0 + 0j         # Slack bus voltage (reference)
V2 = 1.05             # PV bus voltage magnitude (fixed)
PG2 = 0.6661          # Active power at bus 2 (p.u.)
S3 = 2.8653 + 1.2244j  # Load at bus 3: P3=2.8653, Q3=1.2244

# For convenience, treat loads as negative injections:
P3_load = S3.real
Q3_load = S3.imag

# -------------------------------------------------------------------
# Transmission line and shunt:
#   ZL = j0.1
#   YC = j0.01
# -------------------------------------------------------------------
ZL = 0 + j*0.1   # series reactance j0.1
YC = 0 + j*0.01  # shunt admittance j0.01

# -------------------------------------------------------------------
# Admittance Matrix Calculation
#   Y_bus is 3×3 for this fully connected "triangle" of lines
# -------------------------------------------------------------------
Y_bus = np.array([
    [ 2 / ZL + YC,      -1 / ZL,       -1 / ZL     ],
    [     -1 / ZL,  1 / ZL + 1 / ZL + YC,  -1 / ZL ],
    [     -1 / ZL,      -1 / ZL,       2 / ZL + YC ]
], dtype=complex)

###############################################################################
# 2) HELPER FUNCTION: CALCULATE POWER INJECTIONS
###############################################################################
def calc_power_injections(V, Y):
    """
    Given bus voltages V and the admittance matrix Y,
    return complex power injections S = P + jQ.
         S_k = V_k * conj(sum_m [Y_km * V_m])
    """
    I = Y @ V
    S = V * np.conjugate(I)
    return S

###############################################################################
# 3) DECOUPLED NEWTON–RAPHSON POWER FLOW METHOD
###############################################################################
def decoupled_NR_3bus(Y_bus, V2_spec, PG2_spec, P3_load, Q3_load,
                      tol=0.001, max_iter=60, verbose=True):
    """
    Solve the 3-bus power flow using a decoupled Newton–Raphson method.
    
    Bus definitions:
      - Bus 1 (Slack): V1 = 1.0 ∠0° (fixed)
      - Bus 2 (PV):    |V2| = V2_spec (fixed), P2 = PG2_spec (angle unknown)
      - Bus 3 (PQ):    Load: P3_load, Q3_load (both as negative injections)
    
    Unknowns:
      x = [theta2, theta3, V3]
      
    Mismatch equations:
      dP2 = PG2_spec - P_calc[1]
      dP3 = -P3_load  - P_calc[2]
      dQ3 = -Q3_load  - Q_calc[2]
    
    The Jacobian is decoupled as follows:
      - J_Pθ (2×2): partial derivatives of [P2, P3] with respect to [theta2, theta3]
      - J_QV (scalar): partial derivative of Q3 with respect to V3
    
    At each iteration, the method prints:
      - Bus voltage magnitudes and angles
      - Active and reactive power mismatches
      - The decoupled Jacobian entries
      
    Upon convergence, a dictionary is returned containing:
      - Final bus voltages (complex)
      - Bus angles (radians)
      - Active and reactive power injections
      - Final mismatch vector
      - Total number of iterations
      - Final Jacobian blocks
    """
    # Initial guess: [theta2, theta3, V3]
    x = np.array([0.0, 0.0, 1.0], dtype=float)
    h = 1e-6  # finite difference perturbation

    for it in range(max_iter):
        theta2, theta3, V3 = x

        # Convert state to bus voltages
        V1c = 1.0 * np.exp(j*0.0)               # Slack bus
        V2c = V2_spec * np.exp(j*theta2)         # PV bus (angle unknown)
        V3c = V3 * np.exp(j*theta3)              # PQ bus
        V = np.array([V1c, V2c, V3c])

        # Compute power injections
        S_calc = calc_power_injections(V, Y_bus)
        P_calc = S_calc.real
        Q_calc = S_calc.imag

        # Mismatches
        dP2 = PG2_spec - P_calc[1]
        dP3 = -P3_load - P_calc[2]
        dQ3 = -Q3_load - Q_calc[2]
        mismatch = np.array([dP2, dP3, dQ3])
        max_mis = np.max(np.abs(mismatch))

        if verbose:
            print(f"[Decoupled NR] Iter {it+1}:")
            print(f"  theta2 = {theta2:.6f} rad, theta3 = {theta3:.6f} rad, V3 = {V3:.6f} p.u.")
            print(f"  Mismatch: dP2 = {dP2:.6e}, dP3 = {dP3:.6e}, dQ3 = {dQ3:.6e}")
            print("  Bus Voltages:")
            for idx, Vi in enumerate(V):
                print(f"    Bus {idx+1}: |V| = {abs(Vi):.6f} p.u., angle = {np.degrees(np.angle(Vi)):.2f}°")
        
        if max_mis < tol:
            if verbose:
                print(f"Decoupled NR converged in {it+1} iterations.\n")
            break

        # --- Compute decoupled Jacobian entries ---
        # J_Pθ: derivatives of P2 and P3 with respect to theta2 and theta3.
        J_Ptheta = np.zeros((2, 2))
        # Perturb theta2
        x_pert = x.copy()
        x_pert[0] += h
        theta2_p, theta3_p, V3_p = x_pert
        V1_p = 1.0 * np.exp(j*0.0)
        V2_p = V2_spec * np.exp(j*theta2_p)
        V3_p_val = V3_p * np.exp(j*theta3_p)
        V_pert = np.array([V1_p, V2_p, V3_p_val])
        S_pert = calc_power_injections(V_pert, Y_bus)
        P_pert = S_pert.real
        J_Ptheta[0, 0] = - (P_pert[1] - P_calc[1]) / h  # ∂P2/∂theta2
        J_Ptheta[1, 0] = - (P_pert[2] - P_calc[2]) / h  # ∂P3/∂theta2

        # Perturb theta3
        x_pert = x.copy()
        x_pert[1] += h
        theta2_p, theta3_p, V3_p = x_pert
        V1_p = 1.0 * np.exp(j*0.0)
        V2_p = V2_spec * np.exp(j*theta2_p)
        V3_p_val = V3_p * np.exp(j*theta3_p)
        V_pert = np.array([V1_p, V2_p, V3_p_val])
        S_pert = calc_power_injections(V_pert, Y_bus)
        P_pert = S_pert.real
        J_Ptheta[0, 1] = - (P_pert[1] - P_calc[1]) / h  # ∂P2/∂theta3
        J_Ptheta[1, 1] = - (P_pert[2] - P_calc[2]) / h  # ∂P3/∂theta3

        # J_QV: derivative of Q3 with respect to V3
        x_pert = x.copy()
        x_pert[2] += h
        theta2_p, theta3_p, V3_p = x_pert
        V1_p = 1.0 * np.exp(j*0.0)
        V2_p = V2_spec * np.exp(j*theta2_p)
        V3_p_val = V3_p * np.exp(j*theta3_p)
        V_pert = np.array([V1_p, V2_p, V3_p_val])
        S_pert = calc_power_injections(V_pert, Y_bus)
        Q_pert = S_pert.imag
        J_QV = - (Q_pert[2] - Q_calc[2]) / h

        if verbose:
            print("  Jacobian (P–θ block):")
            print(J_Ptheta)
            print(f"  Jacobian (Q–V): {J_QV:.6e}\n")

        # --- Solve for state corrections ---
        # For the P equations: J_Pθ * delta_theta = [dP2, dP3]
        delta_theta = np.linalg.solve(J_Ptheta, np.array([dP2, dP3]))
        # For the Q equation: J_QV * delta_V = dQ3
        delta_V = dQ3 / J_QV

        # Update the state
        x[0] += delta_theta[0]
        x[1] += delta_theta[1]
        x[2] += delta_V

    # End of iterations.
    theta2, theta3, V3 = x
    V1c = 1.0 * np.exp(j*0.0)
    V2c = V2_spec * np.exp(j*theta2)
    V3c = V3 * np.exp(j*theta3)
    V_final = np.array([V1c, V2c, V3c])
    S_final = calc_power_injections(V_final, Y_bus)

    # Prepare final output
    result = {
        'bus_voltages': V_final,
        'bus_angles_rad': np.array([0.0, theta2, theta3]),
        'active_power': S_final.real,
        'reactive_power': S_final.imag,
        'mismatch': mismatch,
        'iterations': it + 1,
        'Jacobian_Ptheta': J_Ptheta,
        'Jacobian_QV': J_QV
    }
    return result

###############################################################################
# 4) MAIN SCRIPT: RUN THE DECOUPLED NR POWER FLOW
###############################################################################
if __name__ == "__main__":
    results = decoupled_NR_3bus(Y_bus, V2, PG2, P3_load, Q3_load,
                                tol=0.001, max_iter=100, verbose=True)
    print("============================================")
    print("Final Results from Decoupled Newton–Raphson:")
    print("============================================")
    for i, V in enumerate(results['bus_voltages']):
        print(f"Bus {i+1}: |V| = {abs(V):.6f} p.u., angle = {np.degrees(np.angle(V)):.2f}°")
    print()
    print("Active Power Injections (p.u.):", results['active_power'])
    print("Reactive Power Injections (p.u.):", results['reactive_power'])
    print("Final Mismatch (p.u.):", results['mismatch'])
    print("Total Iterations:", results['iterations'])
    print("Final Jacobian (P–θ block):")
    print(results['Jacobian_Ptheta'])
    print("Final Jacobian (Q–V entry):", results['Jacobian_QV'])
