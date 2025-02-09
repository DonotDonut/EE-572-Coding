import numpy as np

# Read File
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

# Parse Branch Data
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

# Parse Bus Data
def parse_bus_data(bus_data):
    bus_info = []
    for line in bus_data:
        values = line.split()
        bus_number = int(values[0])
        bus_type = int(values[5])
        PG = float(values[10]) / 100
        QG = float(values[11]) / 100
        Pd = float(values[8]) / 100
        Qd = float(values[9]) / 100
        Vmag = float(values[6])
        delta = float(values[7])
        Qmin = float(values[13]) / 100
        Qmax = float(values[12]) / 100
        
        # Check QG bounds
        QG = max(Qmin, min(QG, Qmax))
        
        bus_info.append((bus_number, bus_type, PG, QG, Pd, Qd, Vmag, delta, Qmin, Qmax))
    return np.array(bus_info)

# Read Input File
file_path0 = 'Term Project/ieee14cdf.txt'
bus_start_marker = "BUS DATA FOLLOWS"
branch_start_marker = "BRANCH DATA FOLLOWS"
stop_marker = "-999"

bus_data, branch_data = read_File(file_path0, bus_start_marker, branch_start_marker, stop_marker)

# Process Data
branch_info = parse_branch_data(branch_data)
bus_info = parse_bus_data(bus_data)

# Check if data was successfully read
if branch_info.size == 0 or bus_info.size == 0:
    raise ValueError("Error: No valid bus or branch data found.")

# Initialize Y-Bus
nbus = int(np.max(branch_info[:, :2]))
Ybus = np.zeros((nbus, nbus), dtype=complex)

# Extract branch parameters
R = branch_info[:, 2].astype(float)
X = branch_info[:, 3].astype(float)
B = 1j * branch_info[:, 4].astype(float)
Z = R + 1j * X
Y_line = 1.0 / Z

nline = len(branch_info)
ng = np.sum(bus_info[:, 1] == 2)

for k in range(nline):
    i = int(branch_info[k, 0]) - 1
    j = int(branch_info[k, 1]) - 1
    Ybus[i, j] -= Y_line[k]
    Ybus[j, i] = Ybus[i, j]
    Ybus[i, i] += Y_line[k] + B[k]
    Ybus[j, j] += Y_line[k] + B[k]

# Extract relevant values
Ymag = np.abs(Ybus)
theta = np.angle(Ybus)

bus_type = bus_info[:, 1].astype(int)
Pg, Qg, Pd, Qd = bus_info[:, 2:6].T
Qmin, Qmax, Vmag, delta = bus_info[:, 8:].T

P_sch = Pg - Pd
Q_sch = Qg - Qd

accuracy = 1.0
iteration = 1
max_iter = 10

while accuracy >= 0.001 and iteration < max_iter:
    P_cal = np.zeros(nbus)
    Q_cal = np.zeros(nbus)

    for i in range(1, nbus):
        for n in range(nbus):
            angle_term = theta[i, n] + delta[n] - delta[i]
            P_cal[i] += Vmag[i] * Vmag[n] * Ymag[i, n] * np.cos(angle_term)
            Q_cal[i] -= Vmag[i] * Vmag[n] * Ymag[i, n] * np.sin(angle_term)

    DP = P_sch[1:] - P_cal[1:]
    PQ_indices = np.where(bus_info[:, 1] == 1)[0]
    DQ = Q_sch[PQ_indices] - Q_cal[PQ_indices]

    DF = np.concatenate((DP, DQ))
    accuracy = np.linalg.norm(DF)
    iteration += 1

# Print results
print("V2 Amplitude (p.u.):", Vmag[1])
print("V2 Angle (degrees):", np.degrees(delta[1]))
print("V3 Angle (degrees):", np.degrees(delta[2]))
