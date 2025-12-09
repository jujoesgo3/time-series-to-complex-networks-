import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

'''1. Se crean los nodos del grafo a partir de la serie de tiempo x en formato
de lista con el tamaño de ventana w definido.''' 

def crear_ind(x, tamaño_w):
    x_w1 = []

    for i in range(len(x)):
        sub = x[i:i+tamaño_w]
        x_w1.append(sub)

    ind_x1 = []
    for i in range(len(x_w1)-tamaño_w):
        ind = [index for index, value in sorted(enumerate(x_w1[i]), key=lambda x: x[1])]
        posiciones = sorted(range(len(ind)), key=lambda k: ind[k])
        ind_x1.append(posiciones)

    return ind_x1


'''2. Se construye el grafo dirigido a partir de los nodos y el tamaño de ventana w'''

def crear_grafo(ind_x1, tamaño_w):
    # Crear un grafo dirigido vacío
    G = nx.DiGraph()

    # Mapa para verificar nodos únicos
    unique_nodes = {}
    edges_added = set()

    # Diccionario para mapear nodos del grafo a sublistas de ind_x1
    nodo_a_sublista = {}

    # Añadir nodos y aristas basados en las particiones ordinales
    for i in range(len(ind_x1)):
        nodo = tuple(ind_x1[i])  # Representar cada sublista como una tupla
        if nodo not in unique_nodes:
            unique_nodes[nodo] = len(unique_nodes)
            nodo_a_sublista[unique_nodes[nodo]] = nodo  # Mapear nodo del grafo a sublista

        if i >= tamaño_w:    
            nodo_prev = tuple(ind_x1[i-tamaño_w])
            edge = (unique_nodes[nodo_prev]+1, unique_nodes[nodo]+1)  # Dirección temporal
            if edge not in edges_added:
                G.add_edge(*edge)
                edges_added.add(edge)

    return G, unique_nodes, nodo_a_sublista


'''3. Se calcula la entropía del grafo dirigido construido'''

def calcular_entropia(unique_nodes, ind_x1, w):
    num_nodos = len(unique_nodes)
    matriz_conectividad = np.zeros((num_nodos, num_nodos), dtype=int)

    # Rellenar la matriz de conectividad dirigida
    for i in range(w, len(ind_x1)):
        nodo_prev = tuple(ind_x1[i-w])
        nodo_actual = tuple(ind_x1[i])
        nodo_prev_index = unique_nodes[nodo_prev]
        nodo_actual_index = unique_nodes[nodo_actual]
        matriz_conectividad[nodo_prev_index, nodo_actual_index] += 1  # Solo dirección temporal

    # Calcular las sumas por fila (salidas)
    sum_m = matriz_conectividad.sum(axis=1)

    # Calcular entropías (mu)
    mu = []
    for i in range(len(sum_m)):
        entropia = 0
        for j in range(len(sum_m)):
            p_ij = matriz_conectividad[i, j] / sum_m[i] if sum_m[i] > 0 else 0
            if p_ij > 0:  # Evitar logaritmos de 0
                entropia -= p_ij * np.log2(p_ij)
        mu.append(entropia)

    mut = sum(mu) / len(mu) if len(mu) > 0 else 0  # Evitar división por cero

    return mu, mut, matriz_conectividad


'''4. Optimización del tamaño de ventana ω'''

def optimizar_w(x, w_min, w_max):
    valores_w = range(w_min, w_max)
    entropias = []

    for w in valores_w:
        # Crear las particiones ordinales (ind_x1) para el valor actual de w
        ind_x1 = crear_ind(x, w)

        # Crear el grafo y obtener los nodos únicos
        _, unique_nodes, _ = crear_grafo(ind_x1, w)

        # Calcular la entropía promedio (mut) para el valor actual de w
        _, mut, _ = calcular_entropia(unique_nodes, ind_x1, w)

        # Almacenar la entropía promedio
        entropias.append(mut)

    # Encontrar el w óptimo (máxima entropía)
    w_optimo = valores_w[np.argmax(entropias)]

    # Graficar la entropía en función de w
    plt.figure(figsize=(8, 5))
    plt.plot(valores_w, entropias, marker='o', linestyle='-', color='black')
    plt.axvline(x=w_optimo, color='r', linestyle='--', label = f'$\omega_o$ = {w_optimo}')
    plt.legend()
    plt.show()

    return w_optimo


def optimizar_w_nogra(x, w_min, w_max):
    valores_w = range(w_min, w_max)
    entropias = []

    for w in valores_w:
        ind_x1 = crear_ind(x, w)
        _, unique_nodes, _ = crear_grafo(ind_x1, w)
        _, mut, _ = calcular_entropia(unique_nodes, ind_x1, w)
        entropias.append(mut)

    w_optimo = valores_w[np.argmax(entropias)]
    return w_optimo

