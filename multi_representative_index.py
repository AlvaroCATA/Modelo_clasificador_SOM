# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 10:55:19 2020

@author: alvaro
"INDICE DE VALIDACIÓN DE REPRESENTACIÓN MÚLTIPLE
"""
from numpy import linalg as LA
import numpy as np
import functions as fun


###################Calculo del CDbw############################################  
"""
Comienza el cálculo del índice de representación múltiple el cual se divide
en varias funcions para generar un código modulado y facil de analizar
"""  
def _calculate_stdev (data,clusters,number_cluster,list_neighbors):
    stdev_vector = np.zeros((number_cluster,data.shape[1]))
    stdev_average = 0
    for i in range (number_cluster):
        stdev_i = 0
        index = np.argwhere(clusters == list_neighbors[i])
        stdev_i = np.std(data[index,:],axis=0)
        stdev_average += (LA.norm(stdev_i))**2
        stdev_vector[i,:] = stdev_i
    stdev_average = np.sqrt((stdev_average/number_cluster))
    return stdev_average,stdev_vector
    
def _intra_density_function (data,weights,clusters,nis,neurons_labels,
                    stdev_average,number_cluster,list_neighbors):    
    intra_den_vec = np.zeros(number_cluster)    
    for i in range(number_cluster):
        acc = 0
        index = np.argwhere(clusters == list_neighbors[i])
        index_ris = np.argwhere(neurons_labels == list_neighbors[i])
        for j in range (len(index_ris)):
            for k in range(nis[i]):
                aux = LA.norm(data[index[k],:] - weights[index_ris[j],:])
                if aux <= stdev_average:
                    acc += 1
        intra_den_vec[i] = acc
    intra_den_c = np.mean(intra_den_vec)
    return intra_den_c,intra_den_vec
        
def _close_representations (data,clusters,number_cluster,list_neighbors):
    matrix_representation_close_i_j = np.zeros ((number_cluster,data.shape[1]))
    for i in range (number_cluster):
        index_i = np.argwhere(clusters == list_neighbors[i]).flatten()
        for j in range(number_cluster):
            if i != j :
                index_j = np.argwhere(clusters == list_neighbors[j]).flatten()
                aux = fun._all_einsum(data[index_i,:],data[index_j,:])
                index_i_j = np.argwhere(aux == np.amin(aux))
                if index_i_j.shape[0] > 1:
                    index_i_j = np.array([index_i_j[0,:]])
                matrix_representation_close_i_j [i,:] = data[index_i[index_i_j[0,0]],:]
    return matrix_representation_close_i_j

def _inter_density_function (data,clusters,nis,number_cluster,stdev_vector,
                    matrix_representation_close_i_j,list_neighbors):
    inter_den_vector = np.zeros((number_cluster))    
    for i in range(number_cluster):
        index_i = np.argwhere(clusters == list_neighbors[i])
        den_1 = LA.norm(stdev_vector[i,:])  
        inter_den = 0
        close_rep_i = matrix_representation_close_i_j[i,:]
        for j in range(number_cluster):
            if i != j:
                den_2 = LA.norm(stdev_vector[j,:])                
                index_j = np.argwhere(clusters == list_neighbors[j])
                close_rep_j = matrix_representation_close_i_j[j,:]
                numerator = LA.norm(close_rep_i-close_rep_j)
                denominator = den_1 + den_2
                u_ij = np.mean([[close_rep_i] ,[close_rep_j]], axis = 0)
                f_uij = 0
                for k in range(nis[i]):
                    aux = LA.norm(data[index_i[k],:]-u_ij)
                    if aux <= ((den_1 + den_2)/2):
                        f_uij += 1
                for k in range(nis[j]):
                    aux = LA.norm(data[index_j[k],:]-u_ij)
                    if aux <= ((den_1 + den_2)/2):
                        f_uij += 1 
                inter_den =+ (numerator/denominator)*f_uij
        inter_den_vector[i] = inter_den
        inter_den = sum(inter_den_vector)
    return inter_den_vector,inter_den

def _sep_function (num_Cluster,matrix_representation_close_i_j,inter_den_vector):
    sep_vector = np.zeros((num_Cluster))    
    for i in range (num_Cluster):
        sep = 0
        close_rep_i = matrix_representation_close_i_j[i,:]
        for j in range (num_Cluster):
            if i != j:
                close_rep_j = matrix_representation_close_i_j[j,:]
                sep  += (LA.norm(close_rep_i-close_rep_j)/(1+inter_den_vector[i]))      
        sep_vector[i] = sep
    sep_c = sum(sep_vector)
    return sep_vector,sep_c


###############################################################################
"""
Retorna el índice de representación múltiple de los grupos
"""   
def _multi_representative_index (Data,clusters,nis,Weights,neurons_labels, num_Cluster):
    stdev_average,stdev_vector = _calculate_stdev (Data,clusters,num_Cluster)   
    intra_den_c,intra_den_vector = _intra_density_function (Data,Weights,clusters,nis,neurons_labels,stdev_average,num_Cluster)
    matrix_representation_close_i_j = _close_representations(Data,clusters,num_Cluster)
    inter_den_vector,inter_den = _inter_density_function (Data,clusters,nis,num_Cluster,stdev_vector,matrix_representation_close_i_j)
    sep_vector,sep_c = _sep_function (Data,clusters,(num_Cluster),matrix_representation_close_i_j,inter_den)
    #print("CDbw",intra_den_c*sep_c)
    return intra_den_c*sep_c