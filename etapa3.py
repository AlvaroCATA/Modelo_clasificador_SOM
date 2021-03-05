# -*- coding: utf-8 -*-
"""
Created on Mon May 18 16:26:04 2020

@author: alvaro
"""
import multi_representative_index as mri
import numpy as np
from scipy import stats
import functions as fun
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
###############################################################################
    
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
            index_min = np.argmin(fun._all_einsum(params_F,params_G))
            groups_check[i,1] = Groups[index_min]
        for i in range(len(groups_check)):
            merge_index = np.argwhere(unique_elements == groups_check[i,1])
            fix_index = np.argwhere(unique_elements == groups_check[i,0])
            nis =[counts_elements[int(merge_index)],counts_elements[int(fix_index)]]
            list_neighboor = np.array([groups_check[i,1],groups_check[i,0]])
            stdev_average,stdev_vector = mri._calculate_stdev (data,labels,2,list_neighboor)     
            intra_den_c,intra_den_vector = mri._intra_density_function (data,weights,labels,
                                                                     nis,neurons_Labels,stdev_average,2,list_neighboor)
            matrix_representation_close_i_j = mri._close_representations(data,labels,2,list_neighboor)
            inter_den_vector,inter_den = mri._inter_density_function (data,labels,nis,2,stdev_vector,
                                                                  matrix_representation_close_i_j,list_neighboor)
            sep_vector,sep_c = mri._sep_function (2,matrix_representation_close_i_j,inter_den_vector)
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
    
    
