import numpy as np

# Initialize Methods 
def read_File(file_path, bus_start_marker, branch_start_marker, stop_marker):
    bus_data = []
    branch_data = []
    reading_bus_data = False
    reading_branch_data = False

    try:
        with open(file_path, 'r') as file:
            for line in file:
                if reading_bus_data:
                    if stop_marker in line:
                        reading_bus_data = False
                        continue
                    bus_data.append(line.strip())
                elif reading_branch_data:
                    if stop_marker in line:
                        reading_branch_data = False
                        continue
                    branch_data.append(line.strip())
                elif bus_start_marker in line:
                    reading_bus_data = True
                elif branch_start_marker in line:
                    reading_branch_data = True
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")

    return bus_data, branch_data

def parse_branch_data(branch_data):
    branch_info = []
    for line in branch_data:
        values = line.split()
        from_bus = int(values[0])
        to_bus = int(values[1])
        resistance = float(values[6])
        reactance = float(values[7])
        line_charging = float(values[8])
        branch_info.append((from_bus, to_bus, resistance, reactance, line_charging))
    return np.array(branch_info)

def parse_bus_data(bus_data):
    bus_info = []
    for line in bus_data:
        values = line.split()
        bus_number = int(values[0])
        
        # Bus Type: 3 = slack, 2 = PV, 1 = PQ, 0 = isolated  
        bus_type = int(values[5])
        
        # All values are in per unit, using 100 as the Sbase value 
        PG = float(values[10]) / 100  # Real Power generated from Generator 
        QG = float(values[11]) / 100  # Reactive Power generated from Generator 
        Pd = float(values[8]) / 100   # Real Power delivered to load  
        Qd = float(values[9]) / 100   # Reactive Power delivered to load 
        Vmag = float(values[6])       # Voltage magnitude
        delta = float(values[7])      # Voltage angle
        Qmin = float(values[15]) / 100 # Min allowed for power generation 
        Qmax = float(values[14]) / 100 # Max allowed for power generation 
        
        bus_info.append((bus_number, bus_type, PG, QG, Pd, Qd, Vmag, delta, Qmin, Qmax))
    return np.array(bus_info)

# Code starts running here (Main Method)
file_path0 = 'Term Project/ieee14cdf.txt'
file_path1 = 'Term Project/ieee30cdf.txt'
file_path2 = 'Term Project/ieee57cdf.txt'
file_path3 = 'Term Project/ieee118cdf.txt' #issue here with DX 
file_path4 = 'Term Project/ieee300cdf.txt' #issue here with DX 

# Markers for reading 
bus_start_marker = "BUS DATA FOLLOWS"
branch_start_marker = "BRANCH DATA FOLLOWS"
stop_marker = "-999"
    
# Reading file 
bus_data, branch_data = read_File(file_path3, bus_start_marker, branch_start_marker, stop_marker)    

# Getting specific data from the branch section in the files text
branch_info = parse_branch_data(branch_data)
'''
# Printing branch data 
print(f"\nBranch Data")
for branch in branch_info:
    print(f"From Bus: {branch[0]}, To Bus: {branch[1]}, Resistance: {branch[2]}, Reactance: {branch[3]}, Line Charging: {branch[4]}")
'''

bus_info = parse_bus_data(bus_data)
'''
# Print extracted bus info
print(f"\nBus Data")
for bus in bus_info:
    print(f"Bus#: {bus[0]}, Type: {bus[1]}, PG: {bus[2]}, QG: {bus[3]}, Pd: {bus[4]}, Qd: {bus[5]}, |V|: {bus[6]}, Delta: {bus[7]}, Qmin: {bus[8]}, Qmax: {bus[9]}")
'''

# Y-Bus Formation
R = branch_info[:, 2]
X = branch_info[:, 3]
B = 1j * branch_info[:, 4]  # Y/2 term multiplied by j
Z = R + 1j * X
Y_line = 1.0 / Z

nline = len(branch_info)
nbus = int(np.max(branch_info[:, :2]))

# Initialize Ybus matrix
Ybus = np.zeros((nbus, nbus), dtype=complex)

for k in range(nline):
    i = int(branch_info[k, 0]) - 1
    j = int(branch_info[k, 1]) - 1

    # Off-diagonal elements
    Ybus[i, j] -= Y_line[k]
    Ybus[j, i] = Ybus[i, j]

    # Diagonal elements
    Ybus[i, i] += Y_line[k] + B[k]
    Ybus[j, j] += Y_line[k] + B[k]

Ymag = np.abs(Ybus)  # Admittance magnitude 
theta = np.angle(Ybus)  # Admittance angle 

# Extract bus data
bus_type = bus_info[:, 1].copy()
Pg = bus_info[:, 2].copy()
Qg = bus_info[:, 3].copy()
Pd = bus_info[:, 4].copy()
Qd = bus_info[:, 5].copy()
Qmin = bus_info[:, 8].copy()
Qmax = bus_info[:, 9].copy()
Vmag = bus_info[:, 6].copy()
delta = bus_info[:, 7].copy()

# Form the complex voltage vector
V = Vmag * (np.cos(delta) + 1j * np.sin(delta))

# Scheduled net injections:
P_sch = Pg - Pd
Q_sch = Qg - Qd

accuracy = 1.0 # edit when needed 
iteration = 1 # edit when needed 
max_iter = 10 # edit when needed 

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
                bus_info[i, 1] = 3  # Temporarily convert PV to PQ
            elif Q_cal[i] < Qmin[i]:
                Q_cal[i] = Qmin[i]
                bus_info[i, 1] = 3  # Temporarily convert PV to PQ
            else:
                bus_info[i, 1] = 2  # Restore PV bus type
                Vmag[i] = bus_info[i, 6]
    
    # Form the mismatch vectors.
    # For P: ignore slack bus (index 0)
    DP = P_sch[1:] - P_cal[1:]
    
    # For Q: use only buses that are PQ (bus type == 3)
    PQ_indices = np.where(bus_info[:, 1] == 3)[0]
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
    non_slack = np.where(bus_info[:, 1] != 1)[0]
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
    
    # Debugging prints
    print("Size of DP:", DP.shape)
    print("Size of DQ:", DQ.shape)
        
    # Ensure correct assembly of J
    print("J11 Shape:", J11.shape)
    print("J22 Shape:", J22.shape)
    print("J33 Shape:", J33.shape)
    print("J44 Shape:", J44.shape)
    
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


# Display Results of Amplitude and Angle of Each Bus
print("\nFinal Bus Voltages:")
print(f"{'Bus#':<5} {'|V| (p.u.)':<12} {'Angle (radians)':<18} {'Angle (degrees)'}")
print("-" * 50)

for n in range(nbus):
    print(f"{n+1:<5} {Vmag[n]:<12.6f} {delta[n]:<18.6f} {np.degrees(delta[n]):.6f}")

