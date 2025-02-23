import numpy as np
from scipy.linalg import lu_factor, lu_solve


class MPOpt:
    def __init__(self, tol, max_it, verbose):
        # Create nested objects for pf options
        self.pf = type('pf', (), {})()
        self.pf.tol = tol
        self.pf.fd = type('fd', (), {})()
        self.pf.fd.max_it = max_it
        self.verbose = verbose

def mpoption():
    # Default options; these may be overridden by user inputs.
    return MPOpt(tol=0.001, max_it=100, verbose=2)

# ---------------------------------------------------------
# Fast Decoupled Power Flow Function (fdpf)
# ---------------------------------------------------------
def fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, mpopt=None):
    
    if mpopt is None:
        mpopt = mpoption()
    
    tol    = mpopt.pf.tol
    max_it = mpopt.pf.fd.max_it

    # Initialize
    converged = False
    i = 0
    V = V0.copy()
    Va = np.angle(V)
    Vm = np.abs(V)
    
    # Set up indexing for updating voltage angles and magnitudes
    indices_pv_pq = np.concatenate((pv, pq))
    
    # Evaluate initial mismatch: mis = (V * conj(Ybus*V) - Sbus) / Vm
    mis = (V * np.conj(Ybus @ V) - Sbus) / Vm
    P = np.real(mis[indices_pv_pq])
    Q = np.imag(mis[pq])
    
    # Check tolerance
    normP = np.linalg.norm(P, np.inf)
    normQ = np.linalg.norm(Q, np.inf)
    if mpopt.verbose > 1:
        print('\niteration     max mismatch (p.u.)')
        print('type   #        P            Q')
        print('---- ----  -----------  -----------')
        print('  -  {:3d}   {:10.3e}   {:10.3e}'.format(i, normP, normQ))
    if normP < tol and normQ < tol:
        converged = True
        if mpopt.verbose > 1:
            print('\nConverged!')
    
    # Reduce B matrices for the required buses
    Bp_reduced = Bp[np.ix_(indices_pv_pq, indices_pv_pq)]
    Bpp_reduced = Bpp[np.ix_(pq, pq)]
    
    # Factor B matrices (LU factorization)
    lu_p, piv = lu_factor(Bp_reduced)
    lu_pp, piv_pp = lu_factor(Bpp_reduced)
    
    # Do P and Q iterations
    while (not converged) and (i < max_it):
        i += 1

        # --- Print Jacobian entries for the current iteration ---
        if mpopt.verbose > 1:
            print("\nIteration {} Jacobian entries:".format(i))
            print("Bp_reduced (for P iteration):")
            print(Bp_reduced)
            print("Bpp_reduced (for Q iteration):")
            print(Bpp_reduced)
        
        # ----- P iteration: update voltage angles (Va) -----
        dVa = -lu_solve((lu_p, piv), P)
        Va[indices_pv_pq] = Va[indices_pv_pq] + dVa
        # Update bus voltages with new angles (voltage magnitudes remain unchanged)
        V = Vm * np.exp(1j * Va)
        
        # Evaluate mismatch after P iteration
        mis = (V * np.conj(Ybus @ V) - Sbus) / Vm
        P = np.real(mis[indices_pv_pq])
        Q = np.imag(mis[pq])
        
        normP = np.linalg.norm(P, np.inf)
        normQ = np.linalg.norm(Q, np.inf)
        if mpopt.verbose > 1:
            print('  P  {:3d}   {:10.3e}   {:10.3e}'.format(i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = True
            if mpopt.verbose:
                print('\nFast-decoupled power flow converged in {} P-iterations and {} Q-iterations.'.format(i, i-1))
            break
        
        # ----- Q iteration: update voltage magnitudes (Vm) -----
        dVm = -lu_solve((lu_pp, piv_pp), Q)
        Vm[pq] = Vm[pq] + dVm
        # Update voltages with new magnitudes
        V = Vm * np.exp(1j * Va)
        
        mis = (V * np.conj(Ybus @ V) - Sbus) / Vm
        P = np.real(mis[indices_pv_pq])
        Q = np.imag(mis[pq])
        
        normP = np.linalg.norm(P, np.inf)
        normQ = np.linalg.norm(Q, np.inf)
        if mpopt.verbose > 1:
            print('  Q  {:3d}   {:10.3e}   {:10.3e}'.format(i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = True
            if mpopt.verbose:
                print('\nFast-decoupled power flow converged in {} P-iterations and {} Q-iterations.'.format(i, i))
            break

    if mpopt.verbose:
        if not converged:
            print('\nFast-decoupled power flow did not converge in {} iterations.'.format(i))
    return V, converged, i

# ---------------------------------------------------------
# Input Data (as provided)
# ---------------------------------------------------------
# Complex unit for convenience
j = 1j

# Bus voltages and generation/load data
V1 = 1.0 + 0j         # Slack bus voltage
V2 = 1.05             # PV bus voltage magnitude
PG2 = 0.6661          # Active power at bus 2 (p.u.)

S3 = 2.8653 + 1.2244j  # Load at bus 3: P3=2.8653, Q3=1.2244
P3_load = S3.real
Q3_load = S3.imag

# Transmission line and shunt parameters
ZL = 0 + j*0.1        # Line impedance (purely reactive)
YC = 0 + j*0.01       # Shunt admittance

# Convergence criteria
max_iter = 100  # maximum iterations
tol = 0.001     # tolerance

# Admittance Matrix Calculation
Y_bus = np.array([
    [ 2 / ZL + YC,      -1 / ZL,       -1 / ZL     ],
    [     -1 / ZL,  1 / ZL + 1 / ZL + YC,  -1 / ZL ],
    [     -1 / ZL,      -1 / ZL,       2 / ZL + YC ]
], dtype=complex)

# ---------------------------------------------------------
# Set up the power flow problem
# ---------------------------------------------------------
# Define Sbus vector for all buses:
#   - Slack bus (bus 1): no specified injection (set to 0)
#   - PV bus (bus 2): specified active power, reactive power not specified (set initial guess to 0)
#   - PQ bus (bus 3): load (negative injection)
Sbus = np.array([0 + 0j, PG2 + 0j, -S3], dtype=complex)

# Initial guess for bus voltages V0 (for slack, PV, and PQ buses)
V0 = np.array([V1, V2 + 0j, 1.0 + 0j], dtype=complex)

# For the fast decoupled method, we require the decoupled matrices Bp and Bpp.
# Here, a simple approximation is to use the negative imaginary part of Y_bus.
Bp = -np.imag(Y_bus)
Bpp = -np.imag(Y_bus)

# Define bus indices (0-indexed)
ref = 0            # Slack bus index (bus 1)
pv = np.array([1]) # PV bus index (bus 2)
pq = np.array([2]) # PQ bus index (bus 3)

# Create an options object with our specified tolerance and max iterations
mpopt_obj = MPOpt(tol=tol, max_it=max_iter, verbose=2)

# ---------------------------------------------------------
# Call the fast decoupled power flow function
# ---------------------------------------------------------
V_final, converged, iterations = fdpf(Y_bus, Sbus, V0, Bp, Bpp, ref, pv, pq, mpopt_obj)

# ---------------------------------------------------------
# Display the final bus voltages and convergence results
# ---------------------------------------------------------
print("\nFinal Bus Voltages:")
for idx, voltage in enumerate(V_final, start=1):
    mag = np.abs(voltage)
    angle_deg = np.degrees(np.angle(voltage))
    print("Bus {}: {:.4f} ∠ {:.2f}°".format(idx, mag, angle_deg))
print("\nConverged:", converged)
print("Total Iterations:", iterations)

# ---------------------------------------------------------
# Additional Results
# ---------------------------------------------------------
# Calculate power flow at each bus: S_calc = V * conj(Y_bus * V)
S_calc = V_final * np.conj(Y_bus @ V_final)
print("\nCalculated Power Flow at each bus (S_calc):")
for idx, s in enumerate(S_calc, start=1):
    print("Bus {}: P = {:.4f} p.u., Q = {:.4f} p.u.".format(idx, s.real, s.imag))

# Compute the mismatch error: (S_calc - Sbus) normalized by voltage magnitude
mismatch = (S_calc - Sbus) / np.abs(V_final)
mismatch_norm = np.linalg.norm(mismatch, np.inf)
print("\nMismatch Error (per bus):")
for idx, m in enumerate(mismatch, start=1):
    print("Bus {}: {:.4e} p.u.".format(idx, abs(m)))
print("Maximum Mismatch Error: {:.4e} p.u.".format(mismatch_norm))
