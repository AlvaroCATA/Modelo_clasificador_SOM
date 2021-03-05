"""
Created on Mon May 18 16:13:51 2020

@author: alvaro
"""
import numpy as np
import copy 
import functions as fun

###############################################################################
"""
Retorna las posiciones de los valores mínimos de distancia de las filas
de la matriz 
"""
def _generate_labels(data,weights):
    labels = np.argmin(fun._all_einsum(data,weights), axis = 1)
    return labels
###############################################################################
    
###############################################################################
"""
Retorna el rango de distancia entre neuronas con base al umbral, tolerancia
y matriz de distancia
"""
def _get_range(distance_matrix,threshold,tolerance):
    D_reshape =  copy.deepcopy(distance_matrix.reshape((distance_matrix.shape[0]*distance_matrix.shape[1])))
    X = distance_matrix.shape[0]
    D = distance_matrix.reshape((distance_matrix.shape[0]*distance_matrix.shape[1]))
    tolerance = tolerance /100
    threshold = threshold /100
    index_max = np.argwhere(D_reshape != 1000.0)
    index_min = np.argwhere(D_reshape != 0.0)
    max_Dis = np.amax(D_reshape[index_max])
    min_Dis = np.amin(D_reshape[index_min])
    index_intersect = np.intersect1d(index_max,index_min)
    for i in range (len(index_intersect)):
        D_reshape[index_intersect[i]] = ((D_reshape[index_intersect[i]]-min_Dis)/(max_Dis-min_Dis))
    index_threshold = np.argwhere((D_reshape > threshold-0.1)&(D_reshape<=threshold))
    index_tolerance = np.argwhere((D_reshape>0)&(D_reshape<=tolerance))
    rango = np.mean(D[index_threshold]) + np.mean(D[index_tolerance])
    D = D.reshape((X,X))
    return rango
###############################################################################

###############################################################################
"""
Retorna la matriz de distancia triangular superior del arreglo introducido
"""       
def _get_distance_matrix(matrix):
    distance_matrix = fun._all_einsum(matrix, matrix)
    distance_matrix[np.tril_indices(matrix.shape[0],k=-1)] = 1000
    return distance_matrix
###############################################################################
    
###############################################################################
"""
Se incia el proceso de división del mapa de neuronas en clases delimitado por
el rango de distancia entre las neuronas raíz y su expansión. Además, se realiza
la doble verificación de neuronas entre la creación del grupo y al finalizar la
primera etapa utilizando el concepto de centroide como punto de referencia
"""
def _create_clusters(distance_matrix,weights,labels,number_of_neurons,
                    size_map_x,size_map_y,threshold):
    Tag_Matrix = np.arange(number_of_neurons).reshape((size_map_x, size_map_y))
    Tag_Matrix_Copy = np.arange(number_of_neurons).reshape((size_map_x, size_map_y))
    cont_0 = 0
    cont_1 = number_of_neurons
    Index_Neurons = np.zeros((1,2))
    Index_Neurons[0,1] = cont_1
    for i in range (Tag_Matrix_Copy.shape[0]):
        for j in range (Tag_Matrix_Copy.shape[1]):
            if Tag_Matrix_Copy [i,j] < number_of_neurons:
                #Se etiqueta la red de neuronas por grupo respectivo
                Neurons = np.argwhere(distance_matrix[cont_0,:]<threshold)
                for k in range (Neurons.shape[0]):
                    index = np.argwhere(Neurons[k] == Tag_Matrix)
                    Tag_Matrix_Copy [index[:,0],index[:,1]] = cont_1
                #Se verifican las neuronas compartidas en dos grupos diferentes
                if i != 0 or j != 0:
                    Index_Neurons = np.append(Index_Neurons,[[cont_0,cont_1]],axis=0)
                    for k in range (Index_Neurons.shape[0]):
                        if Index_Neurons[k,0] != cont_0:
                            Neurons_Copy = np.argwhere(distance_matrix[int(Index_Neurons[k,0]),:]<threshold)
                            Neurons_shared = np.intersect1d(Neurons,Neurons_Copy)
                #Si existen neuronas compartidas se asignan a los grupos correspondientes checando la distancia
                            if Neurons_shared.shape[0]>0:
                                for l in range (Neurons_shared.shape[0]):
                                    if distance_matrix[int(Index_Neurons[k,0]),Neurons_shared[l]]<distance_matrix[cont_0,Neurons_shared[l]]:
                                        index = np.where(Neurons_shared[l] == Tag_Matrix)
                                        Tag_Matrix_Copy [index[0],index[1]] = Index_Neurons[k,1]
                cont_1 += 1
            cont_0 += 1
    cont_1 -= (number_of_neurons-1)
    #Re-etiquetado de la matriz de neuronas
    for i in range (cont_1):
        np.place(Tag_Matrix_Copy,Tag_Matrix_Copy == (number_of_neurons+i),i)
        
    #Obtenermos los centroiodes de los grupos y Se asigna a la neuorona mas
    #cercana de cada centro como la neurona con mas influencia
    Tag_Matrix_Copy = Tag_Matrix_Copy.reshape((size_map_x*size_map_y))
    unique_elements, counts_elements = np.unique(Tag_Matrix_Copy, return_counts=True) 
    centroids = np.zeros((len(unique_elements)), dtype=int)
    for i in range (len(unique_elements)):
        index = np.argwhere(Tag_Matrix_Copy == i)
        centroid = np.mean(weights[index,:],axis = 0)
        centroids[i] = index[np.argmin(fun._all_einsum(centroid, weights[index,:]))]
    #A partir de los centros se verifican de nuevo cuales neuronas que cumplen
    #con la condicion de distancia y threshold para un reetiquetado
    #print(centroids)
    for i in range (len(unique_elements)):
        index_Neurons = np.argwhere(distance_matrix[centroids[i],:]< threshold)
        index =  np.argwhere(Tag_Matrix_Copy == i)
        complement = np.setdiff1d(index_Neurons, index)
        reshape = np.argmin ( fun._all_einsum(weights[complement,:],weights[centroids,:]), axis = 1)
        for j in range (len(complement)):
            Tag_Matrix_Copy[complement[j]] = reshape[j]  
    Tag_Matrix_Copy = Tag_Matrix_Copy.reshape((size_map_x,size_map_y))  
    return Tag_Matrix_Copy
