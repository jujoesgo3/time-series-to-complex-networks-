import networkx as nx
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.duffing_oscillator import simulate_classical, simulate_quantum
from src.network_construction.entropia_grafo import crear_ind, crear_grafo, calcular_entropia, optimizar_w


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data", "generated")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "networks")

def ensure_dirs():
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)


def procesar_combinaciones(x_w, RESULTS_DIR):
    """
    x_w = [
        [w_class, w_quant],
        [x_class, x_quant]
    ]
    """
    x_list = x_w[1]      # [x_class, x_quant]
    w_list = x_w[0]      # [w_class, w_quant]
    nombres = ["class", "quant"]

    for i_x, x in enumerate(x_list):      # x_class, x_quant
        for i_w, w in enumerate(w_list):  # w_class, w_quant

            nombre = f"{nombres[i_x]}_{nombres[i_w]}"
            print(f"\nProcesando combinación: {nombre}")

            # 1) Crear nodos
            nodos = crear_ind(x, w)

            # 2) Crear grafo
            G, unique_nodes, nodo_sulista = crear_grafo(nodos, w)

            # 3) Entropía
            mu, mut, mat = calcular_entropia(unique_nodes, nodos, w)

            # 4) Dibujar grafo
            plt.figure(figsize=(15, 15))
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, node_size=20, node_color="red")
            nx.draw_networkx_edges(G, pos, edge_color="blue")
            nx.draw_networkx_labels(G, pos, {i: str(i) for i in G.nodes()}, font_size=10)
            plt.axis("off")

            # Guardar imagen
            img_path = os.path.join(RESULTS_DIR, f"duffing_{nombre}_network.png")
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Guardar edges
            edges = list(G.edges())
            np.save(os.path.join(RESULTS_DIR, f"duffing_{nombre}_edges.npy"), edges)
	
	    # Guardar listas de nodos
            np.save(os.path.join(RESULTS_DIR, f"duffing_{nombre}_nodos.npy"), nodo_sulista)
            
            # Guardar entropía
            entropy_path = os.path.join(RESULTS_DIR, f"duffing_{nombre}_entropy.npy")
            np.save(entropy_path, np.array([mu, mut, mat], dtype=object))

            print(f" - Guardado:")
            print(f" - Imagen:  {img_path}")
            print(f" - Edges:   duffing_{nombre}_edges.npy")
            print(f" - Entropy: duffing_{nombre}_entropy.npy\n")


def main():
    ensure_dirs()

    # =============================
    # Simulación clásica
    # =============================
    t, y_class = simulate_classical()
    x_class = y_class[0]
    p_class = y_class[1]

    df_class = pd.DataFrame({"t": t, "x": x_class, "p": p_class})
    df_class.to_csv(os.path.join(DATA_DIR, "duffing_class_timeseries.csv"), index=False)

    np.save(os.path.join(DATA_DIR, "duffing_class_states.npy"), y_class)

    # =============================
    # Simulación cuántica
    # =============================
    t, x_t, p_t = simulate_quantum()
    x_quant = x_t
    p_quant = p_t

    df_quant = pd.DataFrame({"t": t, "x": x_quant, "p": p_quant})
    df_quant.to_csv(os.path.join(DATA_DIR, "duffing_quant_timeseries.csv"), index=False)

    np.save(os.path.join(DATA_DIR, "duffing_quant_states.npy"), x_t)

    # =============================
    # Construcción de redes
    # =============================
    w_opt_class = optimizar_w(x_class, w_min=2, w_max=40)
    w_opt_quant = optimizar_w(x_quant, w_min=2, w_max=40)

    x_w = [
        [w_opt_class, w_opt_quant],
        [x_class, x_quant],
    ]

    procesar_combinaciones(x_w, RESULTS_DIR)
    print("\n Proceso Duffing COMPLETADO con éxito.\n")


if __name__ == "__main__":
    main()


