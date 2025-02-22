import numpy as np

###############################################################################
# 1) GIVEN SYSTEM DATA
###############################################################################

# Complex unit for convenience
j = 1j

# -------------------------------------------------------------------
# Bus data:
#   Bus 1: Slack  => V1 = 1.0 + 0j
#   Bus 2: PV     => PG2=0.6661 p.u.,  V2=1.05 p.u.,  Q2 unknown
#   Bus 3: PQ     => S3 = 2.8653 + j1.2244  (Load)
# -------------------------------------------------------------------
V1 = 1.0 + 0j    # Slack bus voltage (reference)
V2 = 1.05        # PV bus voltage magnitude
PG2 = 0.6661     # Active power at bus 2 (p.u.)
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
# 2) HELPER FUNCTIONS
###############################################################################

def calc_power_injections(V, Y):
    """
    Given complex bus voltages V (length N) and the bus admittance matrix Y (N×N),
    return the complex power injection S = P + jQ at each bus:
         S_k = V_k * conj( sum_{m} [Y_km * V_m] )
    """
    I = Y @ V                  # N×1 vector of currents
    S = V * np.conjugate(I)    # S_k = V_k * conj(I_k)
    return S  # array of length N

def line_flows(V):
    """
    Compute line flows and losses for each of the three lines in this 3‑bus system.
    The line current from bus i to j is I_ij = Y_ij*(V_i - V_j).
    Then S_ij = V_i * conj(I_ij).
    
    Returns a dictionary: flows[(i+1, j+1)] = (S_ij, S_ji, S_loss).
    Indices i+1, j+1 just for human-friendly 1-based bus labels.
    """
    flows = {}
    # The 3 lines in a triangle: (1-2), (2-3), (1-3)
    lines = [(0,1), (1,2), (0,2)]
    
    # Off-diagonal line admittance is always -1/ZL in this system:
    Y_line = -1.0 / ZL  # i.e. +j10
    
    for (i,j) in lines:
        # Current from bus i to j:
        I_ij = Y_line*(V[i] - V[j])
        S_ij = V[i]*np.conjugate(I_ij)
        
        # Current from bus j to i:
        I_ji = Y_line*(V[j] - V[i])
        S_ji = V[j]*np.conjugate(I_ji)
        
        # Loss = S_ij + S_ji
        flows[(i+1, j+1)] = (S_ij, S_ji, S_ij + S_ji)
    return flows

###############################################################################
# 3) FULL NEWTON–RAPHSON POWER FLOW
###############################################################################
def newton_raphson_3bus(Y_bus, V2_spec, PG2_spec, P3_load, Q3_load,
                        tol=1e-8, max_iter=50, verbose=True):
    """
    Solve the 3-bus system with:
      Bus1 (Slack): V1=1.0 p.u. ∠ 0° (fixed)
      Bus2 (PV):    P2=PG2_spec,  |V2|=V2_spec,  Q2 unknown
      Bus3 (PQ):    P3_load, Q3_load  (both known, negative injection)
    Unknowns: theta2, theta3, V3
    We'll store state x = [theta2, theta3, V3].
    
    Mismatch equations:
      dP2 = P2_spec  - P2_calc
      dP3 = -P3_load - P3_calc   (load is negative injection)
      dQ3 = -Q3_load - Q3_calc
    """
    # x = [theta2, theta3, V3]
    x = np.array([0.0, 0.0, 1.0], dtype=float)  # initial guess

    def state_to_voltages(x):
        t2, t3, v3 = x
        V1c = 1.0 * np.exp(j*0.0)          # Slack
        V2c = V2_spec * np.exp(j*t2)       # PV bus (fixed magnitude, unknown angle)
        V3c = v3 * np.exp(j*t3)            # PQ bus (unknown magnitude & angle)
        return np.array([V1c, V2c, V3c])

    def mismatches(x):
        V = state_to_voltages(x)
        S_calc = calc_power_injections(V, Y_bus)
        P_calc = S_calc.real
        Q_calc = S_calc.imag
        
        # bus indices: 0->bus1, 1->bus2, 2->bus3
        dP2 = PG2_spec - P_calc[1]
        dP3 = -P3_load - P_calc[2]
        dQ3 = -Q3_load - Q_calc[2]
        
        return np.array([dP2, dP3, dQ3])

    def jacobian_fd(x, h=1e-6):
        """Numerical Jacobian of the mismatch vector w.r.t. x using finite differences."""
        f0 = mismatches(x)
        J = np.zeros((3,3), dtype=float)
        for k in range(3):
            x_pert = x.copy()
            x_pert[k] += h
            f1 = mismatches(x_pert)
            J[:,k] = (f1 - f0)/h
        return J

    for it in range(max_iter):
        F = mismatches(x)
        max_mis = np.max(np.abs(F))
        if verbose:
            print(f"[NR] Iter {it+1}, x={x}, mismatches={F}, max|mis|={max_mis}")
        
        if max_mis < tol:
            if verbose:
                print(f"Newton–Raphson converged in {it+1} iterations.\n")
            break
        
        # Solve for state update
        J = jacobian_fd(x)
        dx = np.linalg.solve(J, -F)
        x += dx
    else:
        print("Newton–Raphson did NOT converge within max_iter.\n")

    return x

###############################################################################
# 4) FAST DECOUPLED POWER FLOW
###############################################################################
def fast_decoupled_3bus(Y_bus, V2_spec, PG2_spec, P3_load, Q3_load,
                        tol=1e-8, max_iter=50, verbose=True):
    """
    Solve the same 3-bus system using the classical Fast Decoupled approach.
    We'll decouple P–θ and Q–V. For a 3‑bus system:
      - Slack bus is bus1 => known angle
      - We solve for theta2, theta3 in the P–θ step
      - We solve for V3 in the Q–V step (bus2's voltage is known, bus3's voltage is unknown)
    """
    # Imag part of Y_bus => B = -Im(Y_bus)
    B_full = -Y_bus.imag
    
    # For P–θ: ignoring shunts, we focus on lines => B' submatrix for buses 2,3
    # Slack bus = bus0 => submatrix is B_full[1:3, 1:3]
    Bp = B_full[1:3, 1:3]
    
    # For Q–V: bus3 is the only PQ bus => submatrix is 1×1 => Bq = B_full[2,2]
    # We'll also typically ignore line charging for B'' but keep it simple here
    Bq = B_full[2,2]
    
    # Initialize unknowns
    theta2 = 0.0
    theta3 = 0.0
    V3 = 1.0

    def calc_p_mismatch(t2, t3, v3):
        """Return [dP2, dP3]."""
        V1c = 1.0*np.exp(j*0.0)
        V2c = V2_spec*np.exp(j*t2)
        V3c = v3*np.exp(j*t3)
        V = np.array([V1c, V2c, V3c])
        S_calc = calc_power_injections(V, Y_bus)
        P_calc = S_calc.real
        
        dP2 = PG2_spec - P_calc[1]
        dP3 = -P3_load - P_calc[2]
        return np.array([dP2, dP3])

    def calc_q_mismatch(t3, v3):
        """Return [dQ3] only, since bus3 is the PQ bus."""
        V1c = 1.0*np.exp(j*0.0)
        V2c = V2_spec*np.exp(j*theta2)
        V3c = v3*np.exp(j*t3)
        V = np.array([V1c, V2c, V3c])
        S_calc = calc_power_injections(V, Y_bus)
        Q_calc = S_calc.imag
        
        dQ3 = -Q3_load - Q_calc[2]
        return np.array([dQ3])

    for it in range(max_iter):
        # --- P–θ update ---
        dP = calc_p_mismatch(theta2, theta3, V3)
        # Solve Bp * dTheta = dP
        dTheta = np.linalg.solve(Bp, dP)
        theta2 += dTheta[0]
        theta3 += dTheta[1]
        
        # --- Q–V update ---
        dQ3 = calc_q_mismatch(theta3, V3)
        # Solve Bq * dV3 = dQ3
        dV3 = dQ3[0] / Bq
        V3 += dV3
        
        max_mis = max(np.max(np.abs(dP)), np.max(np.abs(dQ3)))
        if verbose:
            print(f"[FDPF] Iter {it+1}, dP={dP}, dQ3={dQ3}, "
                  f"theta2={theta2:.5f}, theta3={theta3:.5f}, V3={V3:.5f}, max|mis|={max_mis}")
        
        if max_mis < tol:
            if verbose:
                print("Fast Decoupled PF converged.\n")
            break
    else:
        print("Fast Decoupled PF did NOT converge within max_iter.\n")

    return theta2, theta3, V3

###############################################################################
# 5) DC POWER FLOW
###############################################################################
def dc_power_flow_3bus(Y_bus, PG2_spec, P3_load):
    """
    DC Power Flow assumptions:
      - All |V|=1.0
      - Ignore shunt admittances
      - Slack bus angle = 0
      - Solve angles for bus2, bus3 from real-power balance
    """
    # Build B-matrix from the line susceptances only (ignore shunt j0.01 on diagonal).
    B_full = -Y_bus.imag.copy()
    
    # Each diagonal also has +0.01 from the shunt => remove that for DC
    for i in range(3):
        B_full[i,i] -= 0.01  # remove the shunt portion from diagonal
    
    # We want the submatrix for bus2, bus3 => ignoring slack bus1
    Bsub = B_full[1:3, 1:3]
    
    # Net P injections for bus2, bus3 => bus3 is load => -P3_load
    rhs = np.array([PG2_spec, -P3_load])
    
    # Solve Bsub * [theta2, theta3]^T = rhs
    theta_23 = np.linalg.solve(Bsub, rhs)
    return theta_23[0], theta_23[1]

###############################################################################
# 6) MAIN SCRIPT: RUN & PRINT RESULTS
###############################################################################
if __name__ == "__main__":

    print("============================================")
    print(" 1) FULL NEWTON–RAPHSON POWER FLOW ")
    print("============================================")
    x_NR = newton_raphson_3bus(
        Y_bus, V2, PG2, P3_load, Q3_load,
        tol=1e-8, max_iter=50, verbose=True
    )
    theta2_NR, theta3_NR, V3_NR = x_NR
    # Build final voltage phasors
    V_NR = np.array([
        1.0*np.exp(j*0.0),
        V2*np.exp(j*theta2_NR),
        V3_NR*np.exp(j*theta3_NR)
    ])
    # Print final bus voltages
    print("Final bus voltages (NR method):")
    for b in range(3):
        mag = np.abs(V_NR[b])
        ang = np.angle(V_NR[b], deg=True)
        print(f"  Bus {b+1}: |V|={mag:.4f}, angle={ang:.2f} deg")
    # Line flows
    flows_NR = line_flows(V_NR)
    print("\nLine flows and losses (NR method):")
    for (i,j), (S_ij, S_ji, S_loss) in flows_NR.items():
        print(f"  Line {i}-{j}:")
        print(f"    S_ij = {S_ij.real:.4f} + j{S_ij.imag:.4f} p.u.")
        print(f"    S_ji = {S_ji.real:.4f} + j{S_ji.imag:.4f} p.u.")
        print(f"    Loss = {S_loss.real:.4f} + j{S_loss.imag:.4f} p.u.")
    print()

    print("============================================")
    print(" 2) FAST DECOUPLED POWER FLOW ")
    print("============================================")
    t2_FD, t3_FD, v3_FD = fast_decoupled_3bus(
        Y_bus, V2, PG2, P3_load, Q3_load,
        tol=1e-8, max_iter=50, verbose=True
    )
    V_FD = np.array([
        1.0*np.exp(j*0.0),
        V2*np.exp(j*t2_FD),
        v3_FD*np.exp(j*t3_FD)
    ])
    print("Final bus voltages (FDPF):")
    for b in range(3):
        mag = np.abs(V_FD[b])
        ang = np.angle(V_FD[b], deg=True)
        print(f"  Bus {b+1}: |V|={mag:.4f}, angle={ang:.2f} deg")
    flows_FD = line_flows(V_FD)
    print("\nLine flows and losses (FDPF):")
    for (i,j), (S_ij, S_ji, S_loss) in flows_FD.items():
        print(f"  Line {i}-{j}:")
        print(f"    S_ij = {S_ij.real:.4f} + j{S_ij.imag:.4f} p.u.")
        print(f"    S_ji = {S_ji.real:.4f} + j{S_ji.imag:.4f} p.u.")
        print(f"    Loss = {S_loss.real:.4f} + j{S_loss.imag:.4f} p.u.")
    print()

    print("============================================")
    print(" 3) DC POWER FLOW ")
    print("============================================")
    t2_DC, t3_DC = dc_power_flow_3bus(Y_bus, PG2, P3_load)
    print(f"DC angles (rad): theta2={t2_DC:.6f}, theta3={t3_DC:.6f}")
    print(f"DC angles (deg): theta2={np.degrees(t2_DC):.2f}, theta3={np.degrees(t3_DC):.2f}")
    print("\nDone.")
