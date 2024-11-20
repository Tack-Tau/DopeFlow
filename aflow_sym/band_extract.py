#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt

# Open OUTCAR and read k-point coordiantes 
def read_k_cor(kc_file):
    buff = []
    with open(kc_file) as f:
        for i, line in enumerate(f):
            for j in range(n_kpoints):
                if i == 7 + j*(n_level + 2):
                    buff.append(line.split())
    k_frac = np.reshape(np.array(buff, float), (n_kpoints, 4))
    
    k_cor = np.matmul(reci_lat_array, k_frac.T)
    k_cor = k_cor.T
    return k_cor

# Get k-point position/distance on reciprocal lattice axis
def get_k_dis():
    k_cor = read_k_cor("EIGENVAL")
    k_dis = []
    k_dis_buff = np.linalg.norm( np.array(k_cor[0], float) )
    k_dis.append(k_dis_buff)
    for i in range(n_kpoints-1):
        k_dis_del = np.linalg.norm( np.subtract( np.array(k_cor[i], float), \
                                                 np.array(k_cor[i+1], float) )  )
        k_dis_buff = k_dis_buff + k_dis_del
        k_dis.append(k_dis_buff)
    return k_dis

# Open OUTCAR and read energy levels
def read_e_level(el_file):
    buff = []
    with open(el_file) as f:
        for i, line in enumerate(f):
            for m in range(n_level):
                k = 8 + m
                for j in range(n_kpoints):
                    if i == k + j*(n_level + 2):
                        buff.append(line.split())
    e_level_array = np.reshape(np.array(buff, float), (n_kpoints*n_level, 3))
    e_level_array = np.transpose(e_level_array[:, 1])
    #e_level = e_level_array.tolist()
    e_level = np.reshape(e_level_array, (n_kpoints, n_level))
    return e_level

# Ploting
if __name__ == "__main__":
    # Get reciprocal lattice vectors
    reci_lat_vec1 = input("Please enter reciprocal lattice vector 1: ")
    reci_lat_vec2 = input("Please enter reciprocal lattice vector 2: ")
    reci_lat_vec3 = input("Please enter reciprocal lattice vector 3: ")

    reci_lat_vec1 = np.array(reci_lat_vec1.split(), float)
    reci_lat_vec2 = np.array(reci_lat_vec2.split(), float)
    reci_lat_vec3 = np.array(reci_lat_vec3.split(), float)

    reci_lat_array = np.column_stack(( reci_lat_vec1, reci_lat_vec2, \
                                   reci_lat_vec3, np.zeros((3,), dtype=float) ))
    
    # Get number of energy levels and number of total k-points
    n_level = input("Please enter number of energy levels: ")
    n_kpoints = input("Please enter number of total k-points: ")

    n_level = int(n_level)
    n_kpoints = int(n_kpoints)
    
    # skip the first * kpoints 
    sk = input("Please enter the number of first * k-points you want to skip: ")
    sk = int(sk)
    
    """
    # Get k_dis and e_level
    k_dis = get_k_dis()
    e_level = read_e_level("EIGENVAL")
    for j in range(n_level):
        for i in range(n_kpoints):
            # plt.scatter(k_dis[i], B[i+j])
            plt.plot([k_dis[i], k_dis[i+1]], [e_level[i+j], e_level[i+j+1]], '-b')
    plt.show()
    """
    
    #Write (k_dis, e_level) into file "bands.dat"
    k_dis = get_k_dis()
    e_level = read_e_level("EIGENVAL")
    f = open('bands.dat', 'w')
    for ib in range(n_level):
        for ik in range(n_kpoints-sk):
            f.write("%g   %g\n" % (k_dis[ik], e_level[ik+sk][ib]))
        f.write("\n")
    f.close()
