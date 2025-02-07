import numpy as np

def read_ieee_14_bus_data(filename):
    with open(filename, "r") as file:
        lines = file.readlines()
    
    section = None
    bus_data = []
    branch_data = []

    for line in lines:
        line = line.strip()

        # Detect section headers
        if "BUS DATA FOLLOWS" in line:
            section = "BUS"
            continue
        elif "BRANCH DATA FOLLOWS" in line:
            section = "BRANCH"
            continue
        elif "-999" in line:  # End of section
            section = None
            continue

        # Process bus data
        if section == "BUS":
            parts = line.split()
            if len(parts) >= 5:
                bus_data.append([
                    int(parts[0]),  # Bus Number
                    parts[1],       # Bus Type
                    float(parts[4]),  # Voltage (p.u.)
                    float(parts[5]),  # Angle (deg)
                    float(parts[6]),  # Load (MW)
                    float(parts[7])   # Load (MVar)
                ])

        # Process branch data
        elif section == "BRANCH":
            parts = line.split()
            if len(parts) >= 6:
                branch_data.append([
                    int(parts[0]),  # From Bus
                    int(parts[1]),  # To Bus
                    float(parts[3]),  # Resistance (p.u.)
                    float(parts[4]),  # Reactance (p.u.)
                    float(parts[5])   # Line Charging (p.u.)
                ])
    
    return np.array(bus_data, dtype=float), np.array(branch_data, dtype=float)


# Load data
filename = "ieee14.txt"  # Replace with actual file name
bus_data, branch_data = read_ieee_14_bus_data(filename)

# Y-Bus Formation
R = branch_data[:, 2]
X = branch_data[:, 3]
B = 1j * branch_data[:, 4]

Z = R + 1j * X
Y_line = 1.0 / Z

nbus = int(np.max(branch_data[:, :2]))  # Number of buses
nline = len(branch_data)

# Initialize Ybus Matrix
Ybus = np.zeros((nbus, nbus), dtype=complex)

for k in range(nline):
    i = int(branch_data[k, 0]) - 1
    j = int(branch_data[k, 1]) - 1

    Ybus[i, j] -= Y_line[k]
    Ybus[j, i] = Ybus[i, j]

    Ybus[i, i] += Y_line[k] + B[k]
    Ybus[j, j] += Y_line[k] + B[k]

# Calculate Ybus Magnitude and Angles
Ymag = np.abs(Ybus)
theta = np.angle(Ybus)

# Extract Bus Data
bus_type = bus_data[:, 1].copy()
Pd = bus_data[:, 4]
Qd = bus_data[:, 5]
Vmag = bus_data[:, 2]
delta = np.radians(bus_data[:, 3])

accuracy = 1.0
iteration = 1
max_iter = 20

while accuracy >= 0.001 and iteration < max_iter:
    P_cal = np.zeros(nbus)
    Q_cal = np.zeros(nbus)

    for i in range(1, nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            P_cal[i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
            Q_cal[i] -= Vmag[i] * Vmag[n] * Ymag[i, n] * np.sin(angle_term)
    
    DP = - P_cal[1:]
    PQ_indices = np.where(bus_type == 3)[0]
    DQ = - Q_cal[PQ_indices]

    J1 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            if n != i:
                sin_term = np.sin(theta[i, n] + delta[n] - delta[i])
                J1[i, i] += Vmag[i] * Vmag[n] * Ymag[i, n] * sin_term
                J1[i, n] = -Vmag[i] * Vmag[n] * Ymag[i, n] * sin_term
                J1[n, i] = J1[i, n]

    non_slack = np.where(bus_type != 1)[0]
    J11 = J1[np.ix_(non_slack, non_slack)]

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

    J22 = J2[np.ix_(non_slack, PQ_indices)]

    J3 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            if n != i:
                J3[i, i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
                J3[i, n] = -Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
                J3[n, i] = J3[i, n]

    J33 = J3[np.ix_(PQ_indices, non_slack)]

    J4 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            if n == i:
                J4[i, i] = J4[i, i] - 2 * Vmag[i] * Ymag[i, i] * np.sin(theta[i, i])
            else:
                angle_term = theta[i, n] + delta[n] - delta[i]
                J4[i, i] = J4[i, i] - Vmag[n] * Ymag[i, n] * np.sin(angle_term)

    J44 = J4[np.ix_(PQ_indices, PQ_indices)]

    J = np.block([
        [J11, J22],
        [J33, J44]
    ])

    DF = np.concatenate((DP, DQ))
    DX = np.linalg.solve(J, DF)

    num_non_slack = len(non_slack)
    delta[non_slack] = delta[non_slack] + DX[0:num_non_slack]

    num_pq = len(PQ_indices)
    if num_pq > 0:
        deltaV = DX[num_non_slack:]
        Vmag[PQ_indices] = Vmag[PQ_indices] + deltaV

    accuracy = np.linalg.norm(DF)
    iteration += 1

print(f"Converged in {iteration} iterations")
print("Bus Voltage Magnitudes (p.u.):", Vmag)
print("Bus Voltage Angles (degrees):", np.degrees(delta))
