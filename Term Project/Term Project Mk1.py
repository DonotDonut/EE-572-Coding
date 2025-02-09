# initalize Methods 
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
    return branch_info

def parse_bus_data(bus_data):
    bus_info = []
    for line in bus_data:
        values = line.split()
        bus_number = int(values[0])
        
        #bus Type: 3 = slack, 2 PV, 1 PQ, 0 = isolated  
        bus_type = int(values[5])
        PG = float(values[10]) / 100
        QG = float(values[11]) / 100
        Pd = float(values[8]) / 100
        Qd = float(values[9]) / 100
        voltage_magnitude = float(values[6])
        voltage_angle = float(values[7])
        Qmin = float(values[13]) / 100
        Qmax = float(values[12]) / 100

        # Adding the check for QG
        if QG < Qmin:
            QG = Qmin
        elif QG > Qmax:
            QG = Qmax
        
        bus_info.append((bus_number, bus_type, PG, QG, Pd, Qd, voltage_magnitude, voltage_angle, Qmin, Qmax))
    return bus_info

# Code starts running here AKA main Method 
file_path = 'Term Project/ieee14cdf.txt'
bus_start_marker = "BUS DATA FOLLOWS"
branch_start_marker = "BRANCH DATA FOLLOWS"
stop_marker = "-999"
    
#reading file 
bus_data, branch_data = read_File(file_path, bus_start_marker, branch_start_marker, stop_marker)
    
'''
    # Display the values of Bus and Branch data 
print("Bus Data:")
for data in bus_data:
    print(data)

print("\nBranch Data:")
for data in branch_data:
    print(data)
'''
    
# getting specific data from the branch section in the files text
branch_info = parse_branch_data(branch_data)
'''
# printing branch data 
print(f"\nBranch Data")
for branch in branch_info:
    print(f"From Bus: {branch[0]}, To Bus: {branch[1]}, Resistance: {branch[2]}, Reactance: {branch[3]}, Line Charging: {branch[4]}")
'''

bus_info = parse_bus_data(bus_data)
'''
# Print extracted bus info
for bus in bus_info:
    print(f"Bus #: {bus[0]}, Type: {bus[1]}, PG: {bus[2]}, QG: {bus[3]}, Pd: {bus[4]}, Qd: {bus[5]}, |V|: {bus[6]}, Delta: {bus[7]}, Qmin: {bus[8]}, Qmax: {bus[9]}")
'''
