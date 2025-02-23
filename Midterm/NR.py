# import Libraries 
import numpy as np 
from scipy.linalg import lu_factor, lu_solve

# initalize Method 

# Helper functions ----------------------------------
def cal_power_injections(V,Y): 
    I = Y @ V 
    S = V * np.conjugate(I)
    return S 

def line_flows(V): 
    flows = {} 
    lines = [(0,1), (1,2), (0,2)]
    
    Y_line = -0.1/ZL 
    
    for (i,j) in lines: 
        #current from bus i to j 
        I_ij = Y_line*(V[i] -V[j])
        S_ij = V[i]*np.conjugate(I_ij)
        
        #current from bus j to i 
        I_ji = Y_line*(V[j] -V[i])
        S_ji = V[j]*np.conjugate(I_ji)
        
        # loss = S_ij + S_ji 
        flows[(i+1, j+1)] = (S_ij, S_ji, S_ij + S_ji)
    return flows
        
# Netwn Raphson Power Flow -------------------------------------
def newtown_raphson(Y_bus, V2_spec, PG2_spec, PC_load, Q3_load, tol, max_iter, verbose=True):     

    # x = NR_theta2 , NR_theta3, V3 
    x = np.array([0.0, 0.0, 1.0], dtype=float) # initial guess 
    
    def state_to_voltages(x): 
        t2, t3, V3 = x 
        V1c = 1.0 * np.exp(j*0.0)
        V2c = V2_spec * np.exp(j*t2)
        V3c = V3 * np.exp(j*t3) 
        return np.array([V1c, V2c, V3c])
    
    def mismatches(x): 
        V = state_to_voltages(x)
        S_cal = cal_power_injections(V, Y_bus)
        P_calc = S_cal.real 
        Q_calc = S_cal.imag 
        
        dP2 = PG2_spec - P_calc[1]
        dP3 = -P3_load - P_calc[2]
        dQ3 = -Q3_load - Q_calc[2]

        return np.array([dP2, dP3, dQ3])
    
    def jacobian_fd(x, h=1e-6): 
        
        f0 = mismatches(x)
        J = np.zeros((3,3), dtype=float)
        
        for k in range (3): 
            x_pert = x.copy() 
            x_pert[k] += h 
            f1 = mismatches(x_pert)
            j[:,k] = (f1-f0)/h 
        return J 
    
        for it in range (max_iter): 
            F = mismatches(x)
            max_mis = np.max(np.abs(F))
            
            if verbose: 
                print(f"Iter {it+1}, x={x}, mismatches={F}, max|mis|={max_mis}")
                
            if max_mis < tol: 
                if verbose: 
                    print(f"Netwon Rapshon Converge in {it+1} iteration ")
                    break 
            J = jacobian_fd(x)
            dx = np.linalg.solve(J, -F)
            x += dx 
        else: 
            print("")
                    



# Decoupling Newton-Rapshon (NR) 


# Fast Decoupling Newton-Rapshon (NR) 


#  DC 

# Running Main Code

# all given values are in per unit (p.u) 
j = 1j 
V1_spec = 1.0 + 0j # slack bus 

# PV bus 
V2_spec = 1.05 
PG2_spec = 0.6616 

# PQ Bus 
S3 = 2.8652 + 1.2244j 
P3_load = S3.real
Q3_load = S3.imag

ZL = 0 + j*0.1 # transmission line, reactance 
YC = 0 + j*0.01 # shunt capacitor, admittance 

# convergence criteria 
max_iter = 100 # max iterations 
tol = 0.001 # tolerance 

# admittance maxtrux 
Y_bus = np.array([
    [2 / ZL+YC, -1/ZL, -1/ZL]
    [-1/ZL, 1 / ZL + 1/ ZL+YC -1/ZL],
    [-1/ZL, -1/ZL, 2 / (ZL+YC),],
], dtype = complex)




