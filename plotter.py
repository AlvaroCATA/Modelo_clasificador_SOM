"""
Created on Fri May 22 11:42:12 2020

@author: alvaro
"""
from matplotlib.patches import RegularPolygon
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import cm, colorbar
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from scipy.spatial import distance


colors =  ListedColormap (['b','g','red','c','m','y','k','tan','silver','rosybrown',
'lightcoral','brown','aqua','sienna','darksalmon','orangered','tomato','chocolate',
'peru','bisque','darkorange','lightskyblue','orange','gold','khaki','olive','yellow','deepskyblue',
'lawngreen','darkseagreen','lightgreen','forestgreen','limegreen','lightblue','aquamarine','turquoise',
'darkslategrey','teal','mistyrose','lightblue','yellowgreen','gray','steelblue'])

colors_list = ['b','g','red','c','m','y','k','tan','silver','rosybrown',
'lightcoral','brown','aqua','sienna','darksalmon','orangered','tomato','chocolate',
'peru','bisque','darkorange','lightskyblue','orange','gold','khaki','olive','yellow','deepskyblue',
'lawngreen','darkseagreen','lightgreen','forestgreen','limegreen','lightblue','aquamarine','turquoise',
'darkslategrey','teal','mistyrose','lightblue','yellowgreen','gray','steelblue']

###############################################################################
"""
Función para graficar mapas de topología hexagonal con barra de color
configurable
"""
def _matrix_hex_cmap(matrix,coor_xx,coor_yy,map_size_x,map_size_y,
                         cb_title='set legend',title='set title',cMap=cm.coolwarm):
    edge_color ='gray'
    if cMap == cm.gray:
        edge_color = 'red'
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    ax.set(xlim=(-1.5, map_size_x), ylim=(-1, int(map_size_y*0.9)))
    ax.set_aspect('equal')
    for i in range(map_size_x):
        for j in range(map_size_y):
            wy = coor_yy[(i, j)]*2/np.sqrt(3)*3/4
            hex = RegularPolygon((coor_xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                      facecolor=cMap(matrix[i, j]), alpha=.9, edgecolor=edge_color)
            ax.add_patch(hex)
    divider = make_axes_locatable(plt.gca())
    ax_cb = divider.new_horizontal(size="5%", pad=0.05)    
    cb1 = colorbar.ColorbarBase(ax_cb, cmap=cMap, orientation='vertical', alpha=.9)
    cb1.ax.get_yaxis().labelpad = 20
    cb1.ax.set_ylabel(cb_title,rotation=270, fontsize=20)
    plt.gcf().add_axes(ax_cb)
    ax.set_title(title, loc = 'center',fontsize=20)
###############################################################################
 
###############################################################################
"""
Función para graficar mapas de topología hexagonal
"""
def _matrix_hex(matrix,coor_xx,coor_yy,map_size_x,map_size_y,title='set title'):
    edge_color ='gray'
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    ax.set(xlim=(-1.5, map_size_x), ylim=(-1, int(map_size_y*0.9)))
    ax.set_aspect('equal')
    for i in range(map_size_x):
        for j in range(map_size_y):
            wy = coor_yy[(i, j)]*2/np.sqrt(3)*3/4
            hex = RegularPolygon((coor_xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                      facecolor=colors(int(matrix[i, j])), alpha=.9, edgecolor=edge_color)
            ax.add_patch(hex)

    ax.set_title(title, loc = 'center',fontsize=20)
###############################################################################

###############################################################################
"""
Función para graficar mapas enumerados de topología hexagonal
"""
def _matrix_hex_numbers(matrix,data,labels,coor_xx,coor_yy,map_size_x,map_size_y,title='set title'):
    edge_color ='gray'
    f = plt.figure(figsize=(10,10))
    tags = np.arange(map_size_x*map_size_y).reshape(map_size_x,map_size_y)
    ax = f.add_subplot(111)
    ax.set(xlim=(-1.5, map_size_x), ylim=(-1, int(map_size_y*0.9)))
    ax.set_aspect('equal')
    for i in range(map_size_x):
        for j in range(map_size_y):
            wy = coor_yy[(i, j)]*2/np.sqrt(3)*3/4
            hex = RegularPolygon((coor_xx[(i, j)], wy), numVertices=6, radius=.95/np.sqrt(3),
                      facecolor=colors(int(matrix[i, j])), edgecolor=edge_color)
            ax.add_patch(hex)
    for i in range (data.shape[0]):
        index_m = np.argwhere(tags == labels[i])
        wy = coor_yy[(index_m[0,1]), (index_m[0,0])]*2/np.sqrt(3)*3/4
        plt.text((coor_xx[(index_m[0,1]), (index_m[0,0])])-0.1, ((wy)), int(data[i,data.shape[1]-3]), fontsize=20)
    ax.set_title(title, loc = 'center',fontsize=20)
###############################################################################
    
###############################################################################
"""
Función para graficar la matriz de distancia unificada con topología rectangular
"""
def _matrix_rec(matrix,rows,columns,title='set title'):
    grid = rows
    rows = 2*rows-1
    columns = 2*columns-1
    u_matrix = np.zeros((rows,columns))
    cont = 0
    cont_0 = 0
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    for i in range(rows):
        if (i%2) == 0:
            for j in range (0,columns,2):
                if j+1 != columns:
                    u_matrix[i,j+1] = distance.cdist(matrix[cont,:]-matrix[(cont+1),:], 'euclidean')
                cont += 1
        else:
            for j  in range (0,columns,2):
                u_matrix[i,j] = distance.cdist(matrix[cont_0,:]-matrix[(cont_0+grid),:], 'euclidean')
                cont_0 += 1
    for i in range (1,rows,2):
        for j in range (1,columns,2):
            u_matrix[i,j] = (u_matrix[i,j-1]+u_matrix[i,j+1]+u_matrix[i-1,j]+u_matrix[i+1,j])/4
    ax.imshow(u_matrix,interpolation='none')
    ax.set_title(title, loc = 'center',fontsize=20)  
###############################################################################

###############################################################################
"""
Función para graficar datos bidimensionales a color basado en sus clases
"""
def _2d_plot(data,labels,weights,title='set title'):
    unique, counts = np.unique(weights, return_counts=True) 
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    for i in range (labels.shape[0]):
        data[i,2] = weights[int(labels[i])]
    for i in range(len(unique)):
        index = np.where(data[:,2] == unique[i])
        ax.plot(data[index,0],data[index,1],'o',c = colors_list[i])
    ax.set_xlabel ('X')
    ax.set_ylabel('Y')
    ax.set_title(title, loc = 'center',fontsize=20)  
###############################################################################

###############################################################################    
"""
Función para graficar datos tridimensionales a color basado en sus clases
"""
def _3d_plot(data,labels,weights,title='set title'):
    ax = plt.axes(projection='3d')
    unique, counts = np.unique(weights, return_counts=True)
    for i in range (len(labels)):
        data[i,3] = weights[int(labels[i])]
    for i in range (len(unique)):
        index = np.argwhere(data[:,3] == unique[i])
        xdata = data[index,0]
        ydata = data[index,1]
        zdata = data[index,2]
        ax.scatter(xdata, ydata , zdata,c=colors_list[i], marker='o')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(title,loc = 'center',fontsize=20)
###############################################################################

###############################################################################
"""
Función para graficar la matriz de coeficientes después de entrenar la base
de datos de dígitos (mnist,semeion)
"""
def coeficient_matrix(matrix,size_map_x,size_image):
    f = plt.figure(figsize=(10,10))
    ax = f.add_subplot(111)
    numbers = np.zeros(( ((size_map_x*size_image)+(size_map_x-1)),
                         ((size_map_x*size_image)+(size_map_x-1)) ))   
    init_y = int(numbers.shape[0])
    end_y = int(numbers.shape[0]-size_image)
    init_x = 0
    end_x = size_image
    cont = 0
    for j in range (size_map_x):
        for i in range (size_map_x):
            numbers[end_y:init_y,init_x:end_x] = matrix[cont,:].reshape((size_image,size_image))
            init_y -= (size_image +1)
            end_y -= (size_image + 1)
            cont += 1
        init_y = int(numbers.shape[0])
        end_y = int(numbers.shape[0]-size_image)
        init_x += size_image + 1
        end_x  += size_image+1
    ax.imshow(numbers,interpolation='none',cmap = 'gray_r')
    ax.set_title("Matriz de coeficientes", loc = 'center',fontsize=20)    

    

