import numpy as np


def dc_power_flow(Y_bus, PG2_spec, P3_load):

    # Build B-matrix 
    B_full = -Y_bus.imag.copy()
    
    # remove shunt 
    for i in range(3):
        B_full[i,i] -= 0.01  
    
    Bsub = B_full[1:3, 1:3] 
    
    # Net P injections 
    rhs = np.array([PG2_spec, -P3_load])

    theta = np.linalg.solve(Bsub, rhs)
    return theta[0], theta[1]


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

# convergence criteria 
max_iter = 100 
tol = 0.001 


Y_bus = np.array([
    [ 2 / ZL + YC,               -1 / ZL,      -1 / ZL ],
    [     -1 / ZL,  1 / ZL + 1 / ZL + YC,      -1 / ZL ],
    [     -1 / ZL,               -1 / ZL,  2 / ZL + YC ]
], dtype=complex)



# DC Power Flow Results 
print(" Dc power flow ")
print("============================================")
DC_theta2, DC_theta3 = dc_power_flow(Y_bus, PG2, P3_load)
print(f"DC angles (rad): theta2={DC_theta2:.6f}, theta3={DC_theta3:.6f}")
print(f"DC angles (deg): theta2={np.degrees(DC_theta2):.2f}, theta3={np.degrees(DC_theta3):.2f}")