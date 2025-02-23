import numpy as np



# Helper functions --------------------------------- 

def calc_power_injections(V, Y):
    I = Y @ V                 
    S = V * np.conjugate(I)    
    return S  

def line_flows(V):
    flows = {}
    lines = [(0,1), (1,2), (0,2)] 
    

    Y_line = -1.0 / ZL  
    
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


# Newton Rapshon Power Flow Algorithm -------------------------------------
def newton_raphson_3bus(Y_bus, V2_spec, PG2_spec, P3_load, Q3_load, tol, max_iter, verbose=True):
  
    # x = [theta2, theta3, V3]
    x = np.array([0.0, 0.0, 1.0], dtype=float) # initial guess

    def state_to_voltages(x):
        theta2, theta3, v3 = x
        V1c = 1.0 * np.exp(j*0.0) # Slack
        V2c = V2_spec * np.exp(j*theta2) # PV bus 
        V3c = v3 * np.exp(j*theta3) # PQ bus 
        return np.array([V1c, V2c, V3c])

    def mismatches(x):
        V = state_to_voltages(x)
        S_calc = calc_power_injections(V, Y_bus)
        P_calc = S_calc.real
        Q_calc = S_calc.imag
        
        dP2 = PG2_spec - P_calc[1]
        dP3 = -P3_load - P_calc[2]
        dQ3 = -Q3_load - Q_calc[2]
        
        return np.array([dP2, dP3, dQ3])

    def jacobian_fd(x, h=1e-6):
        
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
            print(f"Iteration: {it+1}, x={x}, mismatches={F}, max|mis|={max_mis}")
        
        if max_mis < tol:
            if verbose:
                print(f"Newton–Raphson converged in {it+1} iterations.\n")
            break
        

        J = jacobian_fd(x)
        dx = np.linalg.solve(J, -F)
        x += dx
    else:
        # note to self: debugging 
        print("Newton–Raphson did NOT converge w/in max iteration.\n")

    return x


# main method starts here  ------------------------------------


j = 1j

V1 = 1.0 + 0j   
V2 = 1.05        
PG2 = 0.6661    

S3 = 2.8653 + 1.2244j  
P3_load = S3.real
Q3_load = S3.imag


ZL = 0 + j*0.1 
YC = 0 + j*0.01 

# convergence criteria 
max_iter = 100 
tol = 0.001 

Y_bus = np.array([
    [ 2 / ZL + YC,               -1 / ZL,      -1 / ZL ],
    [     -1 / ZL,  1 / ZL + 1 / ZL + YC,      -1 / ZL ],
    [     -1 / ZL,               -1 / ZL,  2 / ZL + YC ]
], dtype=complex)


print(" Newton Raphson Power Flow ")
print("============================================")
x_NR = newton_raphson_3bus( Y_bus, V2, PG2, P3_load, Q3_load, tol, max_iter, verbose=True)
theta2_NR, theta3_NR, V3_NR = x_NR  

V_NR = np.array([
        1.0*np.exp(j*0.0),
        V2*np.exp(j*theta2_NR),
        V3_NR*np.exp(j*theta3_NR)
])

print("Final bus voltages:")
for b in range(3):
    mag = np.abs(V_NR[b])
    ang = np.angle(V_NR[b], deg=True)
    print(f"Bus {b+1}: |V|={mag:.4f}, angle={ang:.2f} deg")


flows_NR = line_flows(V_NR)
print("\nLine flows and losses:")
for (i,j), (S_ij, S_ji, S_loss) in flows_NR.items():
    print(f"Line {i}-{j}:")
    print(f"S_ij = {S_ij.real:.4f} + j{S_ij.imag:.4f} p.u.")
    print(f"S_ji = {S_ji.real:.4f} + j{S_ji.imag:.4f} p.u.")
    print(f"Loss = {S_loss.real:.4f} + j{S_loss.imag:.4f} p.u.")
print()