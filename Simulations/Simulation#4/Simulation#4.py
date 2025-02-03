import numpy as np

# Line Data: From, To, R, X, Y/2
linedata = np.array([
    [1, 2, 0, 0.1, 0],
    [1, 3, 0, 0.1, 0],
    [2, 3, 0, 0.1, 0]
])

# Bus Data: Bus No, Bus Type, Pg, Qg, Pd, Qd, |V|, delta, Qmin, Qmax (in per unit)
# Bus Type: Slack(1), PV(2), PQ(3)
# delta = voltage angle 
busdata = np.array([
    #Bus No, Bus Type, Pg, Qg, Pd, Qd, |V|, delta, Qmin, Qmax
    [1,     1,         0,   0,  0,  0,  1,    0, 0, 0],
    [2,     3,         0,   0,  2,  1,  1,    0, 0, 0],
    [3,     2,         0.3, 0,  0,  0,  1,    0, 0, 0]
])

# Constants
nbus = int(np.max(linedata[:, :2]))
nline = len(linedata)

# Admittance Matrix Calculation
Ybus = np.zeros((nbus, nbus), dtype=complex)
Z = linedata[:, 2] + 1j * linedata[:, 3]
Y = 1 / Z
B = 1j * linedata[:, 4]

for k in range(nline):
    Ybus[int(linedata[k, 0]) - 1, int(linedata[k, 1]) - 1] -= Y[k]
    Ybus[int(linedata[k, 1]) - 1, int(linedata[k, 0]) - 1] -= Y[k]
    Ybus[int(linedata[k, 0]) - 1, int(linedata[k, 0]) - 1] += Y[k] + B[k]
    Ybus[int(linedata[k, 1]) - 1, int(linedata[k, 1]) - 1] += Y[k] + B[k]

Ymag = np.abs(Ybus)
theta = np.angle(Ybus)

# Bus Data Extraction
type_ = busdata[:, 1]
Vmag = busdata[:, 6].copy()
delta = np.radians(busdata[:, 7])
Pg = busdata[:, 2]
Qg = busdata[:, 3]
Pd = busdata[:, 4]
Qd = busdata[:, 5]
Qmin = busdata[:, 8]
Qmax = busdata[:, 9]

P_sch = Pg - Pd
Q_sch = Qg - Qd
accuracy = 1
iter = 1

# Iteration Process
while accuracy >= 0.001 and iter < 2:
    P_cal = np.zeros(nbus)
    Q_cal = np.zeros(nbus)
    
    for i in range(1, nbus):
        for n in range(nbus):
            P_cal[i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(theta[i, n] + delta[n] - delta[i])
            Q_cal[i] -= Vmag[i] * Vmag[n] * Ymag[i, n] * np.sin(theta[i, n] + delta[n] - delta[i])
    
    DP = P_sch[1:] - P_cal[1:]
    DQ = Q_sch[np.where(type_ == 3)] - Q_cal[np.where(type_ == 3)]
    
    accuracy = np.linalg.norm(np.hstack((DP, DQ)))
    
    if iter == 1:
        print(f"Iteration {iter}:")
        print(f"V2 Amplitude: {Vmag[1]:.4f}")
        print(f"V2 Angle: {np.degrees(delta[1]):.4f} degrees")
        print(f"V3 Angle: {np.degrees(delta[2]):.4f} degrees")
        break
    
    iter += 1
