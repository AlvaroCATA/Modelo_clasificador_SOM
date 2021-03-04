"""
Created on Mon May 18 16:20:36 2020

@author: alvaro
"""
import numpy as np
from scipy import stats

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
Retorna la matriz de distancia y los parámetros estadísticos 
"""
def _get_params_and_distance_matrix(vector,weights):
    uniques, counts_s = np.unique(vector, return_counts=True) 
    parameters = np.zeros(( len(uniques),(3*weights.shape[1]+1) ))
    for w in range (len(uniques)):
        index = np.argwhere(vector == uniques[w]).flatten()
        parameters[w,0:weights.shape[1]] = np.mean(weights[index,:],axis = 0)
        parameters[w,(weights.shape[1]):(2*weights.shape[1])] = np.std(weights[index,:],axis = 0)
        parameters[w,(2*weights.shape[1]):(3*weights.shape[1])] = stats.mode(weights[index,:])[0].flatten()
        parameters[w,(3*weights.shape[1])] = np.ptp(weights[index,:])
    distance_Parameters = _all_einsum(parameters, parameters) 
    return parameters, distance_Parameters
###############################################################################
    
###############################################################################
"""
Segunda etapa de unión de grupos y creación de prototipos de grupos antes de
la última etapa, la unión está basada en los parámetros estadísticos de los
grupos formados en la primera etapa. Al final se utiliza el concepto de
centroide para reagrupar a las neuronas que no hayan sido bien clasificadas
"""
    
def _merge_and_create_protoclusters (distance_matrix,threshold,weights,
                                    cluster_matrix,statistical_params,
                                    distance,percentage_similarity):
    max_Dis = np.amax(distance)
    gridx = cluster_matrix.shape[0]
    gridy = cluster_matrix.shape[1]
    protoGroups = np.zeros((statistical_params.shape[0]))
    for i in range (statistical_params.shape[0]):
        percentage = (100-percentage_similarity)
        index = np.argwhere(((distance[i,:]*100)/max_Dis) < percentage).flatten().tolist()
        index.remove(i)
        if len(index)>0:
            if len(index) > 1:
                new_index = np.argmin(distance[i,index])
                protoGroups[i] = (index[new_index])
            else:
                protoGroups[i] = index[0]
        else:
            protoGroups[i] = -1
    values, counts = np.unique(protoGroups,return_counts=True) 
    for i in range(len(values)):
        if values[i]>0:
            index = np.argwhere(protoGroups == values[i]).flatten()
            if len(index)>1:
                new_index = np.argmin(distance[int(values[i]),index])
                index_m = np.where(cluster_matrix == int(values[i]))
                cluster_matrix [index_m[0],index_m[1]] = index[new_index]
                protoGroups[protoGroups == index[new_index]] = -1
                protoGroups[protoGroups == int(values[i])] = -1
                values[values == index[new_index]] = -1
                values[i] = -1
            else:
                index_m = np.where(int(values[i]) == cluster_matrix)
                cluster_matrix [index_m[0],index_m[1]] =  index[0]
                protoGroups[protoGroups == index[0]] = -1
                protoGroups[protoGroups == int(values[i])] = -1
                values[values == index[0]] = -1
                values[i] = -1
    values, counts = np.unique(cluster_matrix, return_counts=True) 
    for i in range (len(values)):
        index = np.where(cluster_matrix == int(values[i]))
        cluster_matrix [index[0],index[1]] =i
    #Obtenermos los centroiodes de los grupos y Se asigna a la neurona mas
    #cercana de cada centro como la neurona con mas influencia
    cluster_matrix = cluster_matrix.reshape((gridx*gridy))
    unique_elements, counts_elements = np.unique(cluster_matrix, return_counts=True) 
    centroids = np.zeros((len(unique_elements)), dtype=int)
    for i in range (len(unique_elements)):
        index = np.argwhere(cluster_matrix == i)
        centroid = np.mean(weights[index,:],axis = 0)
        centroids[i] = index[np.argmin(_all_einsum(centroid, weights[index,:]))]
    #A partir de los centros se verifican de nuevo cuales neuronas que cumplen
    #con la condicion de distancia y threshold para un reetiquetado
    for i in range (len(unique_elements)):
        index_Neurons = np.argwhere(distance_matrix[centroids[i],:]< threshold)
        index =  np.argwhere(cluster_matrix == i)
        complement = np.setdiff1d(index_Neurons, index)
        reshape = np.argmin ( _all_einsum(weights[complement,:],weights[centroids,:]), axis = 1)
        for j in range (len(complement)):
            cluster_matrix[complement[j]] = reshape[j]  
    cluster_matrix = cluster_matrix.reshape((gridx,gridy))  
    return cluster_matrix,max_Dis


    
    
