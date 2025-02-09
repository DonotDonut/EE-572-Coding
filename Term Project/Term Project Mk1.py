def read_values_between_markers(file_path, bus_start_marker, branch_start_marker, stop_marker):
   
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

def main():
    file_path = 'Term Project/ieee14cdf.txt'
    bus_start_marker = "BUS DATA FOLLOWS"
    branch_start_marker = "BRANCH DATA FOLLOWS"
    stop_marker = "-999"
    
    #reading file 
    bus_data, branch_data = read_values_between_markers(file_path, bus_start_marker, branch_start_marker, stop_marker)
    
    '''
        # Display the values of Bus and Branch data 
    print("Bus Data:")
    for data in bus_data:
        print(data)

    print("\nBranch Data:")
    for data in branch_data:
        print(data)
    '''
    
    # getting specific data from the files 
    branch_info = parse_branch_data(branch_data)
    
    # printing branch data 
    print(f"\nBranch Data")
    for branch in branch_info:
        print(f"From Bus: {branch[0]}, To Bus: {branch[1]}, Resistance: {branch[2]}, Reactance: {branch[3]}, Line Charging: {branch[4]}")

if __name__ == "__main__":
    main()
