import networkx as nx
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.harmonic_oscillator import simulate_classical
from src.network_construction.entropia_grafo import crear_ind, crear_grafo, calcular_entropia, optimizar_w


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

DATA_DIR = os.path.join(BASE_DIR, "data", "generated")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "networks")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
  
def main():
    ensure_dirs()

    # Simulación
    t, y = simulate_classical()
    x = y[0]
    p = y[1]

    # Guardar series de tiempo
    df = pd.DataFrame({
        "t": t,
        "x": x,
        "p": p
    })
    df.to_csv(os.path.join(DATA_DIR, "harmonic_timeseries.csv"), index=False)
    np.save(os.path.join(DATA_DIR, "harmonic_states.npy"), y)
    # 2. Construir red compleja
    w_opt = optimizar_w(x, w_min=2, w_max=40)

    nodos = crear_ind( x, w_opt)

    G, unique_nodes, nodo_sulista = crear_grafo(nodos, w_opt)

    mu,mut, mat= calcular_entropia(unique_nodes, nodos, w_opt)

    # Dibujar el grafo
    plt.figure(figsize=(15, 15))
    pos = nx.spring_layout(G) # Layout para la visualización
    nx.draw_networkx_nodes(G, pos, node_size=20, node_color="red")
    nx.draw_networkx_edges(G, pos, edge_color="b")
    nx.draw_networkx_labels(G, pos, {i: f"{i}" for i in G.nodes()}, font_size=10)

    plt.axis('off')
    img_path = os.path.join(RESULTS_DIR, "harmonic_network.png")
    plt.savefig(img_path, dpi=300, bbox_inches="tight")

    plt.show()
    
    # 3. Guardar resultados
    edges = list(G.edges())
    np.save(os.path.join(RESULTS_DIR, "harmonic_network_edges.npy"), edges)

    print("Harmonic oscillator: simulación y guardado completados")




if __name__ == "__main__":
    main()

