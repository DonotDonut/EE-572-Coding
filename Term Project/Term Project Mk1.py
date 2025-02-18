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
        
        # Bus Type: 3 = slack, 2 = PV, 0 = PQ
        bus_type = int(values[5])
        
        # All values are in per unit, using 100 as the Sbase value 
        PG = float(values[10]) / 100  # Real Power generated from Generator 
        QG = float(values[11]) / 100  # Reactive Power generated from Generator 
        Pd = float(values[8]) / 100   # Real Power delivered to load  
        Qd = float(values[9]) / 100   # Reactive Power delivered to load 
        Vmag = float(values[6])       # Voltage magnitude
        delta = float(values[7])      # Voltage angle (in radians)
        Qmin = float(values[15]) / 100 # Min allowed for power generation 
        Qmax = float(values[14]) / 100 # Max allowed for power generation 
        
        bus_info.append((bus_number, bus_type, PG, QG, Pd, Qd, Vmag, delta, Qmin, Qmax))
    return np.array(bus_info)

# Code starts running here (Main Method)
file_path0 = 'Term Project/ieee14cdf.txt'
file_path1 = 'Term Project/ieee30cdf.txt'
file_path2 = 'Term Project/ieee57cdf.txt'
file_path3 = 'Term Project/ieee118cdf.txt'  # issue here with DX 
file_path4 = 'Term Project/ieee300cdf.txt'   # issue here with DX 

# Markers for reading 
bus_start_marker = "BUS DATA FOLLOWS"
branch_start_marker = "BRANCH DATA FOLLOWS"
stop_marker = "-999"
    
# Reading file 
bus_data, branch_data = read_File(file_path0, bus_start_marker, branch_start_marker, stop_marker)    

# Getting specific data from the branch section in the files text
branch_info = parse_branch_data(branch_data)
'''
# Printing branch data 
print(f"\nBranch Data")
for branch in branch_info:
    print(f"From Bus: {branch[0]}, To Bus: {branch[1]}, Resistance: {branch[2]}, Reactance: {branch[3]}, Line Charging: {branch[4]}")
#'''

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
B = 1j * branch_info[:, 4]  # Note: Y/2 term multiplied by j

Z = R + 1j * X
Y_line = 1.0 / Z

nline = len(branch_info)
# Bus numbers in linedata are given as 1-indexed numbers.
nbus = int(np.max(branch_info[:, :2]))
ng = np.sum(bus_info[:, 1] == 2)  # number of generator (PV) buses

# Initialize Ybus (Python indexing: 0 ... nbus-1)
Ybus = np.zeros((nbus, nbus), dtype=complex)

for k in range(nline):
    # Convert bus numbers to 0-indexed
    i = int(branch_info[k, 0]) - 1
    j = int(branch_info[k, 1]) - 1

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
# 0: Bus No, 1: Bus Type, 2: PG, 3: QG, 4: Pd, 5: Qd, 6: |V|, 7: delta, 8: Qmin, 9: Qmax
bus_type = bus_info[:, 1].copy()  # Copy to allow modification
Pg = bus_info[:, 2].copy()
Qg = bus_info[:, 3].copy()
Pd = bus_info[:, 4].copy()
Qd = bus_info[:, 5].copy()
Qmin = bus_info[:, 8].copy()
Qmax = bus_info[:, 9].copy()
Vmag = bus_info[:, 6].copy()
delta = bus_info[:, 7].copy()

# Form the complex voltage vector (will be updated during iterations)
V = Vmag * (np.cos(delta) + 1j * np.sin(delta))

# Scheduled net injections:
P_sch = Pg - Pd
Q_sch = Qg - Qd

# Convergence parameters:
mismatch_tol = 0.001  # tolerance for power mismatch norm
epsilon = 1e-6        # tolerance for state variable change (voltage magnitudes and angles)
max_iter = 1         # maximum iterations

accuracy = np.inf
iteration = 1

while iteration <= max_iter:
    # Save current state (angles and voltage magnitudes) for convergence checking:
    state_old = np.concatenate((delta, Vmag))
    
    # Initialize calculated power arrays
    P_cal = np.zeros(nbus)
    Q_cal = np.zeros(nbus)
    
    # Loop over non-slack buses 
    for i in range(1, nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            P_cal[i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
            Q_cal[i] -= Vmag[i] * Vmag[n] * Ymag[i, n] * np.sin(angle_term)
        
        # Q limit checking for PV buses (bus type == 2 in original;
        # if Qmax != 0, then the bus is PV; if limits are violated, convert to PQ (bus type 0))
        if Qmax[i] != 0:
            if Q_cal[i] > Qmax[i]:
                Q_cal[i] = Qmax[i]
                bus_info[i, 1] = 0  # Convert PV to PQ
            elif Q_cal[i] < Qmin[i]:
                Q_cal[i] = Qmin[i]
                bus_info[i, 1] = 0  # Convert PV to PQ
            else:
                bus_info[i, 1] = 2  # Restore PV bus type
                Vmag[i] = bus_info[i, 6]
    
    # Form the mismatch vectors.
    DP = P_sch[1:] - P_cal[1:]
    PQ_indices = np.where(bus_info[:, 1] == 0)[0]
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
    
    non_slack = np.where(bus_info[:, 1] != 3)[0]
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
    
    J33 = J3[np.ix_(PQ_indices, non_slack)]
    
    # J4: dQ/d(V)
    J4 = np.zeros((nbus, nbus))
    for i in range(nbus):
        for n in range(nbus):
            if n == i:
                J4[i, i] -= 2 * Vmag[i] * Ymag[i, i] * np.sin(theta[i, i])
            else:
                angle_term = theta[i, n] + delta[n] - delta[i]
                J4[i, i] -= Vmag[n] * Ymag[i, n] * np.sin(angle_term)
    
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
    
    # Update the complex voltage vector V using updated Vmag and delta
    V = Vmag * (np.cos(delta) + 1j * np.sin(delta))
    
    # Calculate the change in state variables
    state_new = np.concatenate((delta, Vmag))
    state_change = np.linalg.norm(state_new - state_old)
    
    # Compute the mismatch norm (accuracy)
    accuracy = np.linalg.norm(DF)
    #print(f"Iteration {iteration}: mismatch norm = {accuracy:.6f}, state change = {state_change:.6e}")
    
    # Check convergence on state variables:
    if state_change < epsilon:
        print("Convergence reached on voltage magnitude/angle update.")
        break
    
    # Alternatively, if power mismatch is small enough, you might also choose to stop:
    if accuracy < mismatch_tol:
        print("Convergence reached on power mismatch.")
        break

    iteration += 1

print(f"\nFinal iteration: {iteration}")

def wrap_angle(angle_deg):
    """Wrap the angle in degrees to be within -360 to 360."""
    return ((angle_deg + 360) % 720) - 360

# --- Final Output Section ---

# Bus Results with generator outputs
print("\nFinal Bus Voltages and Generator Outputs:")
header = f"{'Bus#':<5} {'|V| (p.u.)':<12} {'Angle (rad)':<14} {'Angle (deg)':<14} {'PG (p.u.)':<12} {'QG (p.u.)':<12}"
print(header)
print("-" * len(header))
for n in range(nbus):
    angle_deg = np.degrees(delta[n])
    angle_deg_wrapped = wrap_angle(angle_deg)
    pg = bus_info[n, 2]
    qg = bus_info[n, 3]
    print(f"{n+1:<5} {Vmag[n]:<12.6f} {delta[n]:<14.6f} {angle_deg_wrapped:<14.6f} {pg:<12.6f} {qg:<12.6f}")

# Branch Current Flow Calculation
print("\nBranch Current Flows:")
branch_header = f"{'Branch':<8} {'From Bus':<10} {'To Bus':<8} {'I (mag, p.u.)':<16} {'I (angle, deg)':<16}"
print(branch_header)
print("-" * len(branch_header))
for k in range(nline):
    i = int(branch_info[k, 0]) - 1
    j = int(branch_info[k, 1]) - 1
    I_branch = (V[i] - V[j]) / Z[k]
    I_mag = np.abs(I_branch)
    I_angle = np.degrees(np.angle(I_branch))
    I_angle_wrapped = wrap_angle(I_angle)
    print(f"{k+1:<8} {i+1:<10} {j+1:<8} {I_mag:<16.6f} {I_angle_wrapped:<16.6f}")
