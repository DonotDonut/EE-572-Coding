import numpy as np


# Line Data 
#   Columns: [From, To, R, X, Y/2] (in perunit)
linedata = np.array([
    [1, 2, 0,   0.1, 0],
], dtype=float)

# Bus Data
#   Columns: [Bus No, Bus Type, Pg, Qg, Pd, Qd, |V|, delta, Qmin, Qmax]
# Bus Types: Slack = 1, PV = 2, PQ = 3
busdata = np.array([
    [1, 1,   0,   0,   0, 0, 1, 0, 0, 0],
    [2, 3,   0,   0,   1.5, 0.5, 1, 0, 0, 0.1],
], dtype=float)


# Y-Bus Formation
R = linedata[:, 2]
X = linedata[:, 3]
B = 1j * linedata[:, 4]  # Note: Y/2 term multiplied by j

Z = R + 1j * X
Y_line = 1.0 / Z

nline = len(linedata)
# Bus numbers in linedata are given as 1-indexed numbers.
nbus = int(np.max(linedata[:, :2]))
ng = np.sum(busdata[:, 1] == 2)  # number of generator (PV) buses

# Initialize Ybus (Python indexing: 0 ... nbus-1)
Ybus = np.zeros((nbus, nbus), dtype=complex)

for k in range(nline):
    # Convert bus numbers to 0-indexed
    i = int(linedata[k, 0]) - 1
    j = int(linedata[k, 1]) - 1

    # Off-diagonal elements
    Ybus[i, j] -= Y_line[k]
    Ybus[j, i]  = Ybus[i, j]

    # Diagonal elements
    Ybus[i, i] += Y_line[k] + B[k]
    Ybus[j, j] += Y_line[k] + B[k]

Ymag = np.abs(Ybus)
theta = np.angle(Ybus)


# Bus Data Collection
# In Python, the columns are:
# 0: Bus No, 1: Bus Type, 2: Pg, 3: Qg, 4: Pd, 5: Qd, 6: |V|, 7: delta, 8: Qmin, 9: Qmax
bus_type = busdata[:, 1].copy()  # Copy to allow modification
Pg = busdata[:, 2].copy()
Qg = busdata[:, 3].copy()
Pd = busdata[:, 4].copy()
Qd = busdata[:, 5].copy()
Qmin = busdata[:, 8].copy()
Qmax = busdata[:, 9].copy()
Vmag = busdata[:, 6].copy()
delta = busdata[:, 7].copy()

# Form the complex voltage vector (not used explicitly later)
V = Vmag * (np.cos(delta) + 1j * np.sin(delta))

# Scheduled net injections:
P_sch = Pg - Pd
Q_sch = Qg - Qd

accuracy = 1.0

# Iteration (Newton-Raphson Loop)
# Does not update the slack bus (bus index 0).
iteration = 1
max_iter = 20

while accuracy >= 0.001 and iteration < max_iter:
    # Initialize calculated power arrays
    P_cal = np.zeros(nbus)
    Q_cal = np.zeros(nbus)
    
    # Loop over non-slack buses (in MATLAB: i=2:nbus; here indices 1...nbus-1)
    for i in range(1, nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            P_cal[i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
            Q_cal[i] -= Vmag[i] * Vmag[n] * Ymag[i, n] * np.sin(angle_term)
        
        # Q limit checking for PV buses (bus type == 2 in original, but here if Qmax != 0)
        if Qmax[i] != 0:
            if Q_cal[i] > Qmax[i]:
                Q_cal[i] = Qmax[i]
                busdata[i, 1] = 3  # Temporarily convert PV to PQ
            elif Q_cal[i] < Qmin[i]:
                Q_cal[i] = Qmin[i]
                busdata[i, 1] = 3  # Temporarily convert PV to PQ
            else:
                busdata[i, 1] = 2  # Restore PV bus type
                Vmag[i] = busdata[i, 6]
    
    # Form the mismatch vectors.
    # For P: ignore slack bus (index 0)
    DP = P_sch[1:] - P_cal[1:]
    
    # For Q: use only buses that are PQ (bus type == 3)
    PQ_indices = np.where(busdata[:, 1] == 3)[0]
    DQ = Q_sch[PQ_indices] - Q_cal[PQ_indices]
    
    
    # Jacobian Matrix Calculation
    # J1: dP/d(delta)
    J1 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            if n != i:
                sin_term = np.sin(theta[i, n] + delta[n] - delta[i])
                J1[i, i] += Vmag[i] * Vmag[n] * Ymag[i, n] * sin_term
                J1[i, n] = - Vmag[i] * Vmag[n] * Ymag[i, n] * sin_term
                # Symmetry: (optional) J1[n,i] = J1[i,n]
                J1[n, i] = J1[i, n]
                
    # Remove slack bus rows and columns (i.e. bus type not equal to 1)
    non_slack = np.where(busdata[:, 1] != 1)[0]
    J11 = J1[np.ix_(non_slack, non_slack)]
    
    # J2: dP/d(V)
    J2 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            if n != i:
                J2[i, i] += Vmag[n] * Ymag[i, n] * np.cos(angle_term)
                J2[i, n] = Vmag[i] * Ymag[i, n] * np.cos(angle_term)
                J2[n, i] = J2[i, n]
            else:
                # For n == i add the self term
                J2[i, i] += 2 * Vmag[i] * Ymag[i, i] * np.cos(theta[i, i])
                
    # For J2, only non-slack rows and PQ (type==3) columns are used:
    J22 = J2[np.ix_(non_slack, PQ_indices)]
    
    # J3: dQ/d(delta)
    J3 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            if n != i:
                J3[i, i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
                J3[i, n] = - Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
                J3[n, i] = J3[i, n]
                
    # For J3, only PQ rows and non-slack columns are used:
    J33 = J3[np.ix_(PQ_indices, non_slack)]
    
    # J4: dQ/d(V)
    J4 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            if n == i:
                J4[i, i] = J4[i, i] - 2 * Vmag[i] * Ymag[i, i] * np.sin(theta[i, i])
            else:
                angle_term = theta[i, n] + delta[n] - delta[i]
                J4[i, i] = J4[i, i] - Vmag[n] * Ymag[i, n] * np.sin(angle_term)
            
    # For J4, only PQ rows and PQ columns are used:
    J44 = J4[np.ix_(PQ_indices, PQ_indices)]
    
    # Assemble the full Jacobian
    J = np.block([
        [J11, J22],
        [J33, J44]
    ])
    

    # Correction Vector Calculation
    DF = np.concatenate((DP, DQ))
    DX = np.linalg.solve(J, DF)
    
    # Update voltage angles (delta) for non-slack buses
    num_non_slack = len(non_slack)
    delta[non_slack] = delta[non_slack] + DX[0:num_non_slack]
    
    # Update voltage magnitudes for PQ buses
    num_pq = len(PQ_indices)
    if num_pq > 0:
        deltaV = DX[num_non_slack:]
        Vmag[PQ_indices] = Vmag[PQ_indices] + deltaV
    
    accuracy = np.linalg.norm(DF)
    iteration += 1



# Display Specific Results
print("V2 Amplitude (p.u.):", Vmag[1])
print("V2 Angle (radians):", delta[1])
print("V2 Angle (degrees):", np.degrees(delta[1]))
