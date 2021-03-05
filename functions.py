"""
Created on Thu Mar  4 18:35:55 2021

@author: alvaro
"""
import copy
import numpy as np
import math as math

###############################################################################
"""
Retorna la matriz de distancia euclidiana entre dos arreglos
"""
def _all_einsum(A,B):
    subts = A[:,None,:] - B
    return np.sqrt(np.einsum('ijk,ijk->ij',subts,subts))
###############################################################################

###############################################################################
"""
Se calcula el error topográfico para los mapas entrenados con topología
hexagonal
"""
def _topographic_error(data,neurons,size_map_y):
    coordinates = np.zeros((neurons.shape[0],2))
    neurons_c = copy.deepcopy(neurons)
    contx=0
    conty=0
    error = 0
    for i in range(coordinates.shape[0]):
        coordinates[i,0] = contx
        coordinates[i,1] = conty
        conty += 1
        if conty == size_map_y:
            contx += 1
            conty = 0
    for i in range (1):#data.shape[0]):
        BMU_1 = np.argmin(_all_einsum(neurons_c,data[i,:]))
        neurons_c[BMU_1,:] = neurons_c[BMU_1,:]+100000
        BMU_2 = np.argmin(_all_einsum(neurons_c,data[i,:]))
        #Check if it is odd or even row
        if (coordinates[BMU_1,1]%2) == 0:
            #print("par")
            if ((coordinates[BMU_1,0] == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]+1) == coordinates[BMU_2,1]) or 
            (((coordinates[BMU_1,0]+1) == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]+1) == coordinates[BMU_2,1])) or 
            (((coordinates[BMU_1,0]+1) == coordinates[BMU_2,0]) and 
            (coordinates[BMU_1,1] == coordinates[BMU_2,1])) or 
            (((coordinates[BMU_1,0]+1) == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]-1) == coordinates[BMU_2,1])) or 
            ((coordinates[BMU_1,0] == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]-1) == coordinates[BMU_2,1])) or 
            (((coordinates[BMU_1,0]-1) == coordinates[BMU_2,0]) and
            (coordinates[BMU_1,1] == coordinates[BMU_2,1]))):
                pass
            else:
                error += 1
                #print("No es vecina")
        else:
            #print("impar")
            if ((coordinates[BMU_1,0]-1 == coordinates[BMU_2,0])
            and ((coordinates[BMU_1,1]+1) == coordinates[BMU_2,1])
            or ((coordinates[BMU_1,0] == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]+1) == coordinates[BMU_2,1])) or 
            (((coordinates[BMU_1,0]+1) == coordinates[BMU_2,0]) and 
            (coordinates[BMU_1,1] == coordinates[BMU_2,1])) or 
            ((coordinates[BMU_1,0] == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]-1) == coordinates[BMU_2,1])) or 
            (((coordinates[BMU_1,0]-1) == coordinates[BMU_2,0]) and 
            ((coordinates[BMU_1,1]-1) == coordinates[BMU_2,1])) or 
            (((coordinates[BMU_1,0]-1) == coordinates[BMU_2,0]) and 
            (coordinates[BMU_1,1] == coordinates[BMU_2,1]))):
                pass
            else:                
                error += 1  
                #print("No es vecina")
        neurons_c[BMU_1,:] = neurons_c[BMU_1,:]-100000
    top_error = error / data.shape[0]
    return top_error
    
###############################################################################

###############################################################################
"""
Retorna la entropía de una matriz
"""
def _get_entropy(matrix):
    unique_elements,counts_elements = np.unique(matrix, return_counts=True)  
    entropy = 0    
    for i in range(len(unique_elements)):
        pi= counts_elements[i]/(matrix.shape[0]*matrix.shape[1])
        #entropy += (pi*(math.log(pi,len(unique_elements))))
        entropy += (pi*(math.log2(pi)))
    entropy = entropy*-1
    return entropy
###############################################################################
    
###############################################################################
    
def _get_image(file_name,M,shiftm,N,shiftn):
    matrix = np.zeros((625,460))
    cluster = np.genfromtxt(file_name,delimiter=',')
    cnt = 0
    init_downup = 0
    end_downup = M
    init_leftright = 0
    end_leftright = N
    lowerBorderImg = int((625-(M-shiftm))/(shiftm))
    for i in range (cluster.shape[0]):
        matrix[init_downup:end_downup,init_leftright:end_leftright] = cluster[cnt]
        init_downup += shiftm
        end_downup += M
        if (i%lowerBorderImg) == 0 and i!=0 :
            init_downup = 0
            end_downup = M
            init_leftright += shiftn
            end_leftright += N
        cnt += 1
    return matrix
