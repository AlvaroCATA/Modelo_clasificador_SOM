"""
Created on Sun Jul 28 10:12:07 2021

@author: alvaro
"""
import functions as fun
from minisom import MiniSom
import numpy as np

"""
Archivos que se crearán durante los 10 entrenamientos con los mapas 
auto-organizados
"""
hit_map = ['hitmap_0.csv','hitmap_1.csv','hitmap_2.csv',
           'hitmap_3.csv','hitmap_4.csv','hitmap_5.csv',
           'hitmap_6.csv','hitmap_7.csv','hitmap_8.csv',
           'hitmap_9.csv']

u_matrix = ['umatrix_0.csv','umatrix_1.csv','umatrix_2.csv',
            'umatrix_3.csv','umatrix_4.csv','umatrix_5.csv',
            'umatrix_6.csv','umatrix_7.csv','umatrix_8.csv',
            'umatrix_9.csv']

u_M_init = ['umatrix_init_0.csv','umatrix_init_1.csv','umatrix_init_2.csv',
            'umatrix_init_3.csv','umatrix_init_4.csv','umatrix_init_5.csv',
            'umatrix_init_6.csv','umatrix_init_7.csv','umatrix_init_8.csv',
            'umatrix_init_9.csv']

dataT = ['data_t_0.csv','data_t_1.csv','data_t_2.csv','data_t_3.csv','data_t_4.csv',
         'data_t_5.csv','data_t_6.csv','data_t_7.csv','data_t_8.csv','data_t_9.csv']

dataV = ['data_v_0.csv','data_v_1.csv','data_v_2.csv','data_v_3.csv','data_v_4.csv',
         'data_v_5.csv','data_v_6.csv','data_v_7.csv','data_v_8.csv','data_v_9.csv']

weights = ['weights_0.csv','weights_1.csv','weights_2.csv','weights_3.csv','weights_4.csv',
           'weights_5.csv','weights_6.csv','weights_7.csv','weights_8.csv','weights_9.csv']

coord_xx = ['coord_xx_0.csv','coord_xx_1.csv','coord_xx_2.csv','coord_xx_3.csv',
               'coord_xx_4.csv','coord_xx_5.csv','coord_xx_6.csv','coord_xx_7.csv',
               'coord_xx_8.csv','coord_xx_9.csv']

coord_yy = ['coord_yy_0.csv','coord_yy_1.csv','coord_yy_2.csv','coord_yy_3.csv',
            'coord_yy_4.csv','coord_yy_5.csv','coord_yy_6.csv','coord_yy_7.csv',
            'coord_yy_8.csv','coord_yy_9.csv']


"""
Se lee la base de datos para agrupar y se configura su estructura
"""
data = np.genfromtxt('Dataset.csv',delimiter =',')
dimensionality = 2
number_of_characteristics = 15
Error_CrossV =  np.zeros((10,2))  
"""
Validación cruzada se selecciona el 80% de los datos de entrada para
el entrenamiento y e 20% para validación para K-FOLD
"""
vector_Training = np.empty((1,3))
vector_Validation = np.empty((1,3))
for i in range (number_of_characteristics):
    index = np.argwhere(data[:,dimensionality] == i+1).flatten()
    # print(len(index))
    get_T_index= np.sort(np.random.choice(index,int(index.shape[0]*0.8),replace = False))
    index[(get_T_index-index[0])] = -1
    get_V_index = index[np.sort(np.array(np.where(index>0)).flatten())]
    #print(len(get_T_index),len(get_V_index))
    vector_Training = np.append(vector_Training,data[get_T_index],axis=0)
    vector_Validation = np.append(vector_Validation,data[get_V_index],axis=0)
T_Set = vector_Training[1:,:]
V_Set = vector_Validation[1:,:]
np.savetxt('data_T.csv',T_Set,delimiter=",")
np.savetxt('data_V.csv',V_Set,delimiter=",")


for i in range (10):
    """
    Inicia el proceso de entranamiento seleccionando el 100% de los datos de
    entrenamiento para armar las k= 10 bases de datos K-FOLD
    """
    vector_Training_k = np.empty((1,3))
    vector_Validation_k = np.empty((1,3))
    for j in range (number_of_characteristics):
        index_k = np.argwhere(T_Set[:,dimensionality] == j+1).flatten()
        get_T_index_k= np.sort(np.random.choice(index_k,int(index_k.shape[0]*0.8),replace = False))
        index_k[(get_T_index_k-index_k[0])] = -1
        get_V_index_k = index_k[np.sort(np.array(np.where(index_k>0)).flatten())]
        vector_Training_k= np.append(vector_Training_k,T_Set[get_T_index_k],axis=0)
        vector_Validation_k = np.append(vector_Validation_k,T_Set[get_V_index_k],axis=0)
    T_Set_k = vector_Training_k[1:,:]
    V_Set_k = vector_Validation_k[1:,:]
    
    """
    Inicia el proceso de normalización de datos, descomentar la primera parte
    si se quiere normalizar por característica o la segunda si se quiere 
    normalizar de forma global
    """
    
    # mins_T = np.amin(T_Set, axis = 1)
    # mins_V = np.amin(V_Set, axis = 1)
    # max_T = np.amax(T_Set, axis = 1)
    # max_V = np.amax(V_Set, axis = 1)
    # for j in range (T_Set.shape[0]):
    #     for k in range (T_Set.shape[1]-1):
    #         T_Set[j,k] = ((T_Set[j,k]-mins_T[j])/(max_T[j]-mins_T[j]))
    # for j in range (V_Set.shape[0]):
    #     for k in range (V_Set.shape[1]-1):
    #         V_Set[j,k] = ( (V_Set[j,k]-mins_V[j])/(max_V[j]-mins_V[j]))

    # mins_T = np.amin(T_Set)
    # mins_V = np.amin(V_Set)
    # max_T = np.amax(T_Set)
    # max_V = np.amax(V_Set)
    # for j in range (T_Set.shape[0]):
    #     for k in range (T_Set.shape[1]-1):
    #         T_Set[j,k] = ((T_Set[j,k]-mins_T)/(max_T-mins_T))
    # for j in range (V_Set.shape[0]):
    #     for k in range (V_Set.shape[1]-1):
    #         V_Set[j,k] = ( (V_Set[j,k]-mins_V)/(max_V-mins_V))
    """
    Parámetros de configuración del mapa auto-organizado
    """
    alpha = [0.01,0.1,0.25] #learning rate     
    size_map_x = 40
    size_map_y = 40
    neighboorhood_radio = size_map_x*0.1
    epochs = T_Set.shape[0]*800
    weights = np.zeros((size_map_x*size_map_y,dimensionality))  
    cont = 0
    """
    Inicia el proceso de entrenamiento y guardado de los mejores mapas
    obtenidos
    """
    som = MiniSom(size_map_x, size_map_y, dimensionality, sigma=neighboorhood_radio,
                 learning_rate = alpha[0], topology='hexagonal',neighborhood_function='gaussian',
                 activation_distance='euclidean',random_seed=None) 
    som.random_weights_init(T_Set_k[:,0:2])
    W_init_umatrix = som.distance_map()
    som.train_random(T_Set_k[:,0:2],epochs,verbose=False)
    for j in range(size_map_x):
        for k in range(size_map_y):
            weights[cont,:]=som.get_weights()[j,k,:]
            cont += 1
    Error_CrossV[i,0] = som.quantization_error(T_Set_k[:,0:2])
    Error_CrossV[i,1] = fun._topographic_error(T_Set_k[:,0:2],weights, size_map_y)
    Wi_0 = W_init_umatrix
    Ac_0 = som.activation_response(T_Set_k[:,0:2])
    W_0 = weights
    UM_0 = som.distance_map()
    Training_0 = T_Set_k
    Validation_0 = V_Set_k
    xx_0,yy_0 = som.get_euclidean_coordinates()
    np.savetxt(u_M_init[i], Wi_0,delimiter=",")
    np.savetxt(hit_map[i],Ac_0,delimiter=',')
    np.savetxt(weights[i],W_0,delimiter = ",")
    np.savetxt(u_matrix[i],UM_0,delimiter = ",")
    np.savetxt(coord_xx[i],xx_0,delimiter = ",")
    np.savetxt(coord_yy[i],yy_0,delimiter = ",")
    np.savetxt(dataT[i],Training_0,delimiter=",")
    np.savetxt(dataV[i],Validation_0,delimiter=',')
    print("termine corrida: ",i)
np.savetxt('Error_CrossV.csv',Error_CrossV,delimiter=',')
