import numpy as np
from scipy.linalg import lu_factor, lu_solve

def fdpf(Ybus, Sbus, V0, Bp, Bpp, ref, pv, pq, tol, max_it, verbose):
    converged = False
    i = 0
    V = V0.copy()
    Va = np.angle(V)
    Vm = np.abs(V)
    

    indices_pv_pq = np.concatenate((pv, pq))
    

    mis = (V * np.conjugate(Ybus @ V) - Sbus) / Vm
    P = np.real(mis[indices_pv_pq])
    Q = np.imag(mis[pq])
    
    normP = np.linalg.norm(P, np.inf)
    normQ = np.linalg.norm(Q, np.inf)

    if normP < tol and normQ < tol:
        converged = True
        if verbose > 1:
            print('\nConverged!')
    

    Bp_reduced = Bp[np.ix_(indices_pv_pq, indices_pv_pq)]
    Bpp_reduced = Bpp[np.ix_(pq, pq)]
    

    lu_p, piv = lu_factor(Bp_reduced)
    lu_pp, piv_pp = lu_factor(Bpp_reduced)
    

    while (not converged) and (i < max_it):
        i += 1
        
        if verbose > 1:
            print("\nIteration {} Jacobian entries:".format(i))
            print("Bp_reduced (for P iteration):")
            print(Bp_reduced)
            print("Bpp_reduced (for Q iteration):")
            print(Bpp_reduced)
        
        dVa = -lu_solve((lu_p, piv), P)
        Va[indices_pv_pq] += dVa
        V = Vm * np.exp(1j * Va)
 
        mis = (V * np.conjugate(Ybus @ V) - Sbus) / Vm
        P = np.real(mis[indices_pv_pq])
        Q = np.imag(mis[pq])
        
        normP = np.linalg.norm(P, np.inf)
        normQ = np.linalg.norm(Q, np.inf)
        if verbose > 1:
            print('P {:3d} {:10.3e} {:10.3e}'.format(i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = True
            break

        dVm = -lu_solve((lu_pp, piv_pp), Q)
        Vm[pq] += dVm
        V = Vm * np.exp(1j * Va)
        
        mis = (V * np.conjugate(Ybus @ V) - Sbus) / Vm
        P = np.real(mis[indices_pv_pq])
        Q = np.imag(mis[pq])
        
        normP = np.linalg.norm(P, np.inf)
        normQ = np.linalg.norm(Q, np.inf)
        if verbose > 1:
            print('Q {:3d} {:10.3e} {:10.3e}'.format(i, normP, normQ))
        if normP < tol and normQ < tol:
            converged = True
            break

    if verbose:
        if not converged:
            print('\n Did not converge within max iterations.')
    return V, converged, i

# main method starts here  ------------------------------------


j = 1j

V1 = 1.0 + 0j       
V2 = 1.05            
PG2 = 0.6661         

S3 = 2.8653 + 1.2244j 
P3_load = S3.real
Q3_load = S3.imag

ZL = 0 + j * 0.1     
YC = 0 + j * 0.01    

max_iter = 100       
tol = 0.001         


Y_bus = np.array([
    [ 2 / ZL + YC,               -1 / ZL,      -1 / ZL ],
    [     -1 / ZL,  1 / ZL + 1 / ZL + YC,      -1 / ZL ],
    [     -1 / ZL,               -1 / ZL,  2 / ZL + YC ]
], dtype=complex)


Sbus = np.array([0 + 0j, PG2 + 0j, -S3], dtype=complex)


V0 = np.array([V1, V2 + 0j, 1.0 + 0j], dtype=complex)


Bp = -np.imag(Y_bus)
Bpp = -np.imag(Y_bus)

ref = 0            
pv = np.array([1]) 
pq = np.array([2]) 


V_final, converged, iterations = fdpf(Y_bus, Sbus, V0, Bp, Bpp, ref, pv, pq, tol, max_iter, verbose=2)


print("\nFinal Bus Voltages:")
for b in range(3):
    mag = np.abs(V_final[b])
    ang = np.angle(V_final[b], deg=True)
    print(f"Bus {b+1}: {mag:.4f}  {ang:.2f}°")

print("\nConverged:", converged)
print("Total Iterations:", iterations)

S_calc = V_final * np.conjugate(Y_bus @ V_final)
print("\nCalculated Power Flow at each bus (S_calc):")
for b in range(3):
    print(f"Bus {b+1}: P = {S_calc[b].real:.4f} p.u., Q = {S_calc[b].imag:.4f} p.u.")

mismatch = (S_calc - Sbus) / np.abs(V_final)
mismatch_norm = np.linalg.norm(mismatch, np.inf)
print("\nMismatch Error (per bus):")
for b in range(3):
    print(f"Bus {b+1}: {abs(mismatch[b]):.4e} p.u.")
print(f"Maximum Mismatch Error: {mismatch_norm:.4e} p.u.")