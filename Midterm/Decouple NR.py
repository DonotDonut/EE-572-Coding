import numpy as np

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

max_iter = 100     
tol = 0.001        

Y_bus = np.array([
    [2 / ZL + YC,      -1 / ZL,       -1 / ZL],
    [-1 / ZL,  1 / ZL + 1 / ZL + YC,      -1 / ZL],
    [-1 / ZL,         -1 / ZL, 2 / ZL + YC]
], dtype=complex)

n = Y_bus.shape[0]
Vm = np.array([np.abs(V1), V2, 1.0])
Delta = np.array([np.angle(V1), 0.0, 0.0])


pa = np.array([1, 2])  
pv = np.array([2])     
o = len(pa)
l = len(pv)


Pg = np.array([0.0, PG2, 0.0])
Pl = np.array([0.0, 0.0, P3_load])
Qg = np.array([0.0, 0.0, 0.0])
Ql = np.array([0.0, 0.0, Q3_load])
P = Pg - Pl  
Q = Qg - Ql  


Ym = np.abs(Y_bus)
YTheta = np.angle(Y_bus)

Iter = 0
while True:
    F = np.zeros(n)
    for idx in pa:
        SumP = 0
        for t in range(n):
            SumP += Vm[idx] * Vm[t] * Ym[idx, t] * np.cos(YTheta[idx, t] - Delta[idx] + Delta[t])
        F[idx] = SumP

    G = np.zeros(n)
    for idx in pv:
        SumQ = 0
        for t in range(n):
            SumQ -= Vm[idx] * Vm[t] * Ym[idx, t] * np.sin(YTheta[idx, t] - Delta[idx] + Delta[t])
        G[idx] = SumQ

    dP = P[pa] - F[pa]
    dQ = Q[pv] - G[pv]
    Pm = np.concatenate((dP, dQ))
    
    H = np.zeros((n, n))
    for i in pa:
        for k in pa:
            if i != k:
                H[i, k] = -Vm[i] * Vm[k] * Ym[i, k] * np.sin(YTheta[i, k] - Delta[i] + Delta[k])
            else:
                sumH = 0
                for t in range(n):
                    if t != i:
                        sumH += Vm[i] * Vm[t] * Ym[i, t] * np.sin(YTheta[i, t] - Delta[i] + Delta[t])
                H[i, i] = sumH

    N = np.zeros((n, n))
    for u in pv:
        for v in pv:
            if u != v:
                N[u, v] = -Vm[u] * Ym[u, v] * np.sin(YTheta[u, v] - Delta[u] + Delta[v])
            else:
                sumN = 0
                for t in range(n):
                    if t != u:
                        sumN += Vm[t] * Ym[u, t] * np.sin(YTheta[u, t] - Delta[u] + Delta[t])
                N[u, u] = -sumN - 2 * Vm[u] * Ym[u, u] * np.sin(YTheta[u, u])
    
    L = np.zeros((o, l))
    M = np.zeros((l, o))
    H_sub = H[np.ix_(pa, pa)]
    N_sub = N[np.ix_(pv, pv)]
    J = np.block([[H_sub, L],
                  [M,    N_sub]])
    
    ea = np.max(np.abs(Pm))
    print(f"Iteration {Iter+1}:")
    print("Mismatch vector:", Pm)
    print("Max mismatch error: {:.6f}".format(ea))
    print("Jacobian matrix:")
    print(J)
    print("-" * 60)
    
    try:
        R = np.linalg.solve(J, Pm)
    except np.linalg.LinAlgError:
        print("Jacobian is singular. Stopping iterations.")
        break

    RAngle = R[0:o]
    RVoltage = R[o:o+l]
    for k, idx in enumerate(pa):
        Delta[idx] += RAngle[k]
    for k, idx in enumerate(pv):
        Vm[idx] += RVoltage[k]

    Iter += 1
    if Iter >= max_iter or ea <= tol:
        break

V_NR = np.array([Vm[i] * np.exp(1j * Delta[i]) for i in range(n)])

print("\nFinal bus voltages (NR method):")
for b in range(n):
    mag = np.abs(V_NR[b])
    ang = np.angle(V_NR[b], deg=True)
    print(f"  Bus {b+1}: {mag:.4f}  {ang:.2f}°")

def line_flows(V):
    flows = {}
    for i in range(n):
        for j in range(i+1, n):
            if np.isclose(Y_bus[i, j], -1/ZL):
                S_ij = V[i] * np.conjugate((V[i] - V[j]) / ZL)
                S_ji = V[j] * np.conjugate((V[j] - V[i]) / ZL)
                S_loss = S_ij + S_ji
                flows[(i+1, j+1)] = (S_ij, S_ji, S_loss)
    return flows

flows_NR = line_flows(V_NR)
print("\nLine flows and losses:")
for (i, j), (S_ij, S_ji, S_loss) in flows_NR.items():
    print(f"  Line {i}-{j}:")
    print(f"    S_ij = {S_ij.real:.4f} + j{S_ij.imag:.4f} p.u.")
    print(f"    S_ji = {S_ji.real:.4f} + j{S_ji.imag:.4f} p.u.")
    print(f"    Loss = {S_loss.real:.4f} + j{S_loss.imag:.4f} p.u.")

print(f"\nTotal iterations: {Iter}")
print(f"Final mismatch error: {ea:.6f}")