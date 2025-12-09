import networkx as nx
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys

# Agregar src al PATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.kuramoto_model import simulate_classical, simulate_quantum
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
    x_list = x_w[1]
    w_list = x_w[0]
    nombres = ["class", "quant"]

    for i_x, x in enumerate(x_list):
        for i_w, w in enumerate(w_list):

            nombre = f"{nombres[i_x]}_{nombres[i_w]}"
            print(f"\nProcesando combinación: {nombre}")

            # 1) Crear nodos
            nodos = crear_ind(x, w)

            # 2) Crear grafo
            G, unique_nodes, nodo_sulista = crear_grafo(nodos, w)

            # 3) Entropía
            mu, mut, mat = calcular_entropia(unique_nodes, nodos, w)

            # 4) Graficar
            plt.figure(figsize=(15, 15))
            pos = nx.spring_layout(G)
            nx.draw_networkx_nodes(G, pos, node_size=20, node_color="red")
            nx.draw_networkx_edges(G, pos, edge_color="blue")
            nx.draw_networkx_labels(G, pos, {i: str(i) for i in G.nodes()}, font_size=10)
            plt.axis("off")

            img_path = os.path.join(RESULTS_DIR, f"kuramoto_{nombre}_network.png")
            plt.savefig(img_path, dpi=300, bbox_inches="tight")
            plt.close()

            # Guardar edges
            np.save(os.path.join(RESULTS_DIR, f"kuramoto_{nombre}_edges.npy"), list(G.edges()))
            
            # Guardar listas de nodos
            np.save(os.path.join(RESULTS_DIR, f"Kuramoto_{nombre}_nodos.npy"), nodo_sulista)

            # Guardar entropía
            np.save(os.path.join(RESULTS_DIR, f"kuramoto_{nombre}_entropy.npy"),
                    np.array([mu, mut, mat], dtype=object))

            print(f" - Imagen:  {img_path}")
            print(f" - Edges:   kuramoto_{nombre}_edges.npy")
            print(f" - Entropy: kuramoto_{nombre}_entropy.npy\n")


def main():
    ensure_dirs()

    # =============================
    # Simulación clásica
    # =============================
    t, theta, r = simulate_classical()

    x_class = r
    #p_class = theta

    df_class = pd.DataFrame({"t": t, "r": x_class})
    df_class.to_csv(os.path.join(DATA_DIR, "kuramoto_class_timeseries.csv"), index=False)

    np.save(os.path.join(DATA_DIR, "kuramoto_class_states.npy"),
            np.array([r], dtype=object))

    # =============================
    # Simulación cuántica
    # =============================
    tlist, theta_q, n_t, r_semi, angle_semi = simulate_quantum()

    x_quant = r_semi
    #p_quant = angle_semi

    df_quant = pd.DataFrame({"t": tlist, "r": x_quant})
    df_quant.to_csv(os.path.join(DATA_DIR, "kuramoto_quant_timeseries.csv"), index=False)

    np.save(os.path.join(DATA_DIR, "kuramoto_quant_states.npy"),
            np.array([r_semi], dtype=object))

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

    print("\n Proceso Kuramoto COMPLETADO con éxito.\n")


if __name__ == "__main__":
    main()

