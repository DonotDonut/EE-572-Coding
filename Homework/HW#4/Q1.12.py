import numpy as np

# Line Data (for a 2-bus system)
#   Columns: [From, To, R, X, Y/2] (in perunit)
linedata = np.array([
    [1, 2, 0,   0.1, 0]  # One line connecting slack bus (1) to PQ bus (2)
], dtype=float)

# Bus Data
#   Columns: [Bus No, Bus Type, Pg, Qg, Pd, Qd, |V|, delta, Qmin, Qmax]
# Bus Types: Slack = 1, PQ = 3
busdata = np.array([
    [1, 1,   0,   0,   0, 0, 1, 0, 0, 0],  # Slack Bus
    [2, 3,   0,   0,   1.5, 0.5, 1, 0, 0, 0]   # PQ Bus
], dtype=float)

# Y-Bus Formation
R = linedata[:, 2]
X = linedata[:, 3]
B = 1j * linedata[:, 4]  # Note: Y/2 term multiplied by j

Z = R + 1j * X
Y_line = 1.0 / Z

nline = len(linedata)
nbus = int(np.max(linedata[:, :2]))  # There are 2 buses in this case

# Initialize Ybus (Python indexing: 0 ... nbus-1)
Ybus = np.zeros((nbus, nbus), dtype=complex)

for k in range(nline):
    i = int(linedata[k, 0]) - 1
    j = int(linedata[k, 1]) - 1

    Ybus[i, j] -= Y_line[k]
    Ybus[j, i]  = Ybus[i, j]

    Ybus[i, i] += Y_line[k] + B[k]
    Ybus[j, j] += Y_line[k] + B[k]

Ymag = np.abs(Ybus)
theta = np.angle(Ybus)

# Bus Data Collection
bus_type = busdata[:, 1].copy()  # Copy to allow modification
Pg = busdata[:, 2].copy()
Qg = busdata[:, 3].copy()
Pd = busdata[:, 4].copy()
Qd = busdata[:, 5].copy()
Qmin = busdata[:, 8].copy()
Qmax = busdata[:, 9].copy()
Vmag = busdata[:, 6].copy()
delta = busdata[:, 7].copy()

# Scheduled net injections
P_sch = Pg - Pd
Q_sch = Qg - Qd

accuracy = 1.0

# Iteration (Newton-Raphson Loop)
iteration = 1
max_iter = 20

while accuracy >= 0.001 and iteration < max_iter:
    # Initialize calculated power arrays
    P_cal = np.zeros(nbus)
    Q_cal = np.zeros(nbus)

    # Loop over non-slack buses (i.e., only PQ bus in this case)
    for i in range(1, nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            P_cal[i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
            Q_cal[i] -= Vmag[i] * Vmag[n] * Ymag[i, n] * np.sin(angle_term)
    
    # Form the mismatch vectors.
    DP = P_sch[1:] - P_cal[1:]

    # For Q, use only PQ buses (bus type == 3)
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
                J1[i, n] = -Vmag[i] * Vmag[n] * Ymag[i, n] * sin_term
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
                J3[i, n] = -Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
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
