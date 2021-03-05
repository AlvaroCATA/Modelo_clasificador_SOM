# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:26:04 2020

@author: alvaro
"""

import numpy as np
from scipy import stats
from numpy import linalg as LA

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
Retorna los estadísticos de los grupos creados
"""
def get_statistical_descriptors(index,weights):
    parameters = np.zeros((1,(3*weights.shape[1]+1) ))
    parameters[0,0:weights.shape[1]] = np.mean(weights[index,:],axis = 0)
    parameters[0,(weights.shape[1]):(2*weights.shape[1])] = np.std(weights[index,:],axis = 0)
    parameters[0,(2*weights.shape[1]):(3*weights.shape[1])] = stats.mode(weights[index,:])[0].flatten()
    parameters[0,(3*weights.shape[1])] = np.ptp(weights[index,:])
    return parameters

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
                aux = _all_einsum(data[index_i,:],data[index_j,:])
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
Tercera y última etapa en la creación de clases en el mapa de neuronas. Las
uniones se llevan a cabo calculando el índice de representación múltiple entre
los grupos uniendo a los grupos con menor índice lo cual índica que tienen una
mala separabilidad y que se encuentran muy dispersos sus datos en el grupo. Los 
grupos bases son elegidos utlizando el mapa de activación de la SOM
"""
def _create_classes(distance_matrix, threshold, data, weights, labels, 
                  cluster_matrix, number_groups, hitmap, answer,fixed_groups):
    #Se etiquetan los datos de entrada con base en la matriz etiquetada 
     #             creada en la fun merge_and_Create_Protoclusters
    labels = labels + (cluster_matrix.shape[0]*cluster_matrix.shape[1])
    #print(Tag_Matrix.shape[0]*Tag_Matrix.shape[1])
    neurons_Labels = cluster_matrix.reshape((cluster_matrix.shape[0]*cluster_matrix.shape[1])) 
    for i in range (len(neurons_Labels)):
        index = np.argwhere(labels == (cluster_matrix.shape[0]*cluster_matrix.shape[1]+i))
        labels[index] = neurons_Labels[i]
    #print(np.unique(Labels))
    #Se definen las listas, counts, elements que se utilizaran en el ciclo de merge
    unique_elements, counts_elements = np.unique(labels, return_counts=True)    
    values, counts = np.unique(neurons_Labels, return_counts=True)
    #print(unique_elements,values)
    limit_Groups = np.array(np.where(counts != np.amax(counts))).flatten()
    max_Counts = (np.amax(counts)+1)*10
    Groups = []
    neurons_Alone = []
    fix_Groups = []
    groups =[]
    CDbw = []
    #Se obtiene la lista de los grupos en la red y neuronas no asociadas
    while(len(limit_Groups)>0):
        index_Min = np.argwhere(unique_elements == np.argmin(counts))
        if len(index_Min) == 0:
            neurons_Alone.append(int(np.argmin(counts)))
        else:
            Groups.append(int(np.argmin(counts)))
        counts[np.argmin(counts)] = max_Counts
        limit_Groups =  np.argwhere(counts != np.amax(counts))
    if answer == 'n' or answer == 'N':    
        for i in range (len(Groups)):
            index = np.argwhere(neurons_Labels == Groups[i])
            #*******************************importante******************************
            groups.append(np.sum(hitmap[index]))
            #groups.append(len(hitmap[index])/len(index))
            #groups.append(np.amax(hitmap[index])/len(index))
        while(len(fix_Groups) < number_groups):
            index = np.argmax(groups)
            fix_Groups.append(Groups[index])
            Groups.remove(Groups[index])
            groups.remove(np.amax(groups))
    else:
        fix_Groups = fixed_groups
        for i in range (len(fix_Groups)):
            Groups.remove(fixed_groups[i])
    print(fix_Groups,Groups)
    #length = len(Groups)
    length_dynamic = len(Groups)
    groups_check = np.zeros((len(fix_Groups),2))
    #Inicia el proceso de calcular el CDbw de todos los grupos pequeños 
    #contra todos los grupos grandes excluyendo a las neuronas que no 
    #tienen asociado ningun vector de entrada para realizar el merge
    while(len(Groups)>0):
        CDbw.clear()
        for i in range (len(fix_Groups)):
            index_F = np.argwhere(neurons_Labels == fix_Groups[i])
            groups_check[i,0] = fix_Groups[i]
            params_F = get_statistical_descriptors(index_F,weights)
            params_G = np.zeros((length_dynamic,((3*weights.shape[1])+1)))
            for j in range (length_dynamic):
                index_G = np.argwhere(neurons_Labels == Groups[j])
                params_G[j,:] = get_statistical_descriptors(index_G,weights)
            index_min = np.argmin(_all_einsum(params_F,params_G))
            groups_check[i,1] = Groups[index_min]
        for i in range(len(groups_check)):
            merge_index = np.argwhere(unique_elements == groups_check[i,1])
            fix_index = np.argwhere(unique_elements == groups_check[i,0])
            nis =[counts_elements[int(merge_index)],counts_elements[int(fix_index)]]
            list_neighboor = np.array([groups_check[i,1],groups_check[i,0]])
            stdev_average,stdev_vector = _calculate_stdev (data,labels,2,list_neighboor)     
            intra_den_c,intra_den_vector = _intra_density_function (data,weights,labels,
                                                                     nis,neurons_Labels,stdev_average,2,list_neighboor)
            matrix_representation_close_i_j = _close_representations(data,labels,2,list_neighboor)
            inter_den_vector,inter_den = _inter_density_function (data,labels,nis,2,stdev_vector,
                                                                  matrix_representation_close_i_j,list_neighboor)
            sep_vector,sep_c = _sep_function (2,matrix_representation_close_i_j,inter_den_vector)
            CDbw.append(intra_den_c*sep_c)
        #Comienza el proceso de merge a los grupos con menor CDbw entre ellos
        min_CDbw = np.argmin(CDbw)
        merge = groups_check[min_CDbw,1]
        fix = groups_check[min_CDbw,0]        
        length_dynamic -= 1
        #print("Grupos a mezclar",fix,merge,length_dynamic)
        index_Labels = np.array(np.where(labels == merge)).flatten()
        index_Neurons = np.array(np.where(neurons_Labels == merge)).flatten()
        labels[index_Labels] = fix
        neurons_Labels[index_Neurons] = fix
        Groups.remove(merge)
        unique_elements, counts_elements = np.unique(labels, return_counts=True)
    unique_elements, counts_elements = np.unique(neurons_Labels, return_counts=True)
    new_Groups = unique_elements
    #Se etiquetan las neuronas no asociadas con valores negativos a el número
    #de grupos formados en un inicio para etiquetar a los demás grupos
    for i in range (len(neurons_Alone)):
        element = int(np.argwhere(new_Groups == neurons_Alone[i]))
        new_Groups=np.delete(new_Groups,element)
        index = np.array(np.where(neurons_Labels == neurons_Alone[i])).flatten()
        neurons_Labels[index] = -(i+ max_Counts)
    #Se etiquetan los grupos mezclados
    for i in range (len(new_Groups)):
        index = np.array(np.where(neurons_Labels == new_Groups[i])).flatten()
        neurons_Labels[index] = -(i*10)
    unique_elements, counts_elements = np.unique(neurons_Labels, return_counts=True)
    neurons_Labels *= -1
    for i in range (len(neurons_Alone)):
        index = np.argwhere(neurons_Labels == (i+ max_Counts))
        neurons_Labels[index] = -(i+ max_Counts)
    #Se re-etiquetan las neuronas no asignadas siguiendo con la lógica de etiquetado
    #empleada para los grupos grandes
    max_tag = np.amax(neurons_Labels)
    #print(max_tag)
    for i in range (len(neurons_Alone)):
        index = np.argwhere(neurons_Labels == -(max_Counts+i))
        neurons_Labels [index] = ((i+1)*10) + max_tag 
    #Obtenermos los centroiodes de los grupos y Se asigna a la neuorona mas
    #cercana de cada centro como la neurona con mas influencia
    #Se regresa en forma de matriz el vector de neuronas etiquetadas
    merge_Matrix = neurons_Labels.reshape((cluster_matrix.shape[0],cluster_matrix.shape[1]))
    
    #merge_Matrix = neurons_Labels.reshape((Tag_Matrix.shape[0],Tag_Matrix.shape[1]))
    return merge_Matrix   
    
###############################################################################
    
    
