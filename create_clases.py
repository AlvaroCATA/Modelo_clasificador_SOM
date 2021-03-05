# -*- coding: utf-8 -*-
"""
Created on Sun Jul 14 10:12:07 2019

@author: alvaro
"""
import numpy as np
import copy 
import etapa1
import etapa2
import etapa3

data_t = 'data_t_0.csv'
hit_mapT = 'hitmap_0.csv'
um_Init ='Umatrix_init_0.csv'
xx_T = 'coord_xx_0.csv'
yy_T = 'coord_yy_0.csv'
weights_T = 'weights_0.csv'
umatrix_T = 'umatrix_0.csv'
stage_1T = 'create_groups.csv'
stage_2T = 'create_and_merge_0.csv'
stage_3T = 'create_classes_0.csv'
    
#******************************************************************************
groups = 15
size_map_x = 40
size_map_y = 40
number_of_neurons = 1600
dimensionality = 2
# Se lee la base de datos original, para entrenar y validación
Data = np.genfromtxt(data_t,delimiter =',')
hit_map = np.genfromtxt(hit_mapT,delimiter=',')
Weights = np.genfromtxt(weights_T,delimiter=',')
xx = np.genfromtxt(xx_T,delimiter = ',')
yy = np.genfromtxt(yy_T,delimiter = ',')
umatrix = np.genfromtxt(umatrix_T,delimiter=',')
stage_1 = stage_1T
stage_2 = stage_2T
stage_3 = stage_3T


labels = etapa1._generate_labels(Data[:,0:dimensionality],Weights)
D = etapa1._get_distance_matrix(Weights)
rango = etapa1._get_range(D,10,10)

clusters = etapa1._create_clusters(D,Weights,labels,number_of_neurons,size_map_x,size_map_y,rango)
np.savetxt(stage_1,clusters,delimiter=',')
print("primera etapa")
proto_groups = copy.deepcopy(clusters)
statistical_params, distance = etapa2._get_params_and_distance_matrix(proto_groups.reshape((number_of_neurons)),Weights)

proto_groups,max_dis = etapa2._merge_and_create_protoclusters (D,rango,Weights,proto_groups,statistical_params,distance,80)
np.savetxt(stage_2,proto_groups,delimiter=',')
classes = copy.deepcopy(proto_groups)
print("Segunda etapa")

merge_Matrix = etapa3._create_classes(D,rango,Data[:,0:dimensionality],Weights, 
                labels,classes,groups,hit_map.reshape((number_of_neurons)), 
                answer='n',fixed_Groups=[])
np.savetxt(stage_3,merge_Matrix,delimiter=',')        
print("Tercera etapa")

