# Author:  Jinkai Wang
import networkx as nx
import random
import os

def directed_random_walk_sampling(graph, seed_nodes, walk_length=10, sample_size=20000, p_restart=0.7):
    """
    Effectue un échantillonnage par marche aléatoire améliorée sur un graphe orienté.
    - Évite de rester bloqué sur des nœuds sans successeurs.
    - Permet une probabilité de redémarrage pour sauter vers un nouveau nœud de départ.

    :param graph: Le graphe orienté (NetworkX DiGraph).
    :param seed_nodes: Liste des nœuds initiaux pour démarrer les marches aléatoires.
    :param walk_length: Longueur maximale de chaque marche aléatoire.
    :param sample_size: Nombre total de nœuds à échantillonner.
    :param p_restart: Probabilité de redémarrage à un nœud de départ.
    :return: Sous-graphe échantillonné.
    """
    sampled_nodes = set(seed_nodes)

    for node in seed_nodes:
        current_node = node
        for _ in range(walk_length):
            if random.random() < p_restart:
                current_node = random.choice(seed_nodes)  # Avec une certaine probabilité, redémarre à un nœud de départ.
            else:
                neighbors = list(graph.successors(current_node))  # Considère uniquement les successeurs (arcs sortants).
                if not neighbors:
                    break  # Si aucun successeur, termine la marche actuelle.
                current_node = random.choice(neighbors)

            sampled_nodes.add(current_node)
            if len(sampled_nodes) >= sample_size:
                break
        if len(sampled_nodes) >= sample_size:
            break

    return graph.subgraph(sampled_nodes)


def save_triplets_to_txt(edges, file_path="../data_sampled/KG_train_sampled.txt"):
    """
    Sauvegarde les triplets (arêtes) du graphe dans un fichier texte.

    :param edges: Liste des arêtes sous forme de triplets (tête, relation, queue).
    :param file_path: Chemin du fichier de sortie.
    """
    with open(file_path, "w", encoding="utf-8") as f:
        for head, relation, tail in edges:
            f.write(f"{head}\t{relation}\t{tail}\n")


def KG_sampling(kg_path, walk_length=10, sample_size=20000, p_restart=0.7, num_hubs=300, output_folder="."):
    """
    Effectue un échantillonnage d'un sous-graphe à partir d'un graphe de connaissances.

    :param kg_path: Chemin du fichier contenant le graphe de connaissances.
    :param walk_length: Longueur maximale de chaque marche aléatoire.
    :param sample_size: Nombre total de nœuds à échantillonner.
    :param p_restart: Probabilité de redémarrage à un nœud de départ.
    :param num_hubs: Nombre de nœuds avec le plus grand degré sortant à utiliser comme hubs.
    :return: Sous-graphe échantillonné.
    """
    # Lecture du graphe de connaissances
    G = nx.read_edgelist(os.path.join(kg_path, 'KG_train.txt'), create_using=nx.DiGraph(), nodetype=str, data=[("relation", str)])
    
    # Sélection des nœuds hubs (ceux avec les plus hauts degrés sortants)
    high_outdegree_nodes = sorted(G.out_degree, key=lambda x: x[1], reverse=True)[:num_hubs]
    hub_seeds = [node for node, _ in high_outdegree_nodes]
    # print(hub_seeds)
    
    # Sélection des nœuds valides (ceux ayant au moins un successeur)
    valid_seeds = [node for node in G.nodes() if len(list(G.successors(node))) > 0]
    
    # Combinaison des hubs et d'un échantillon aléatoire de nœuds valides
    seed_nodes1 = random.sample(valid_seeds, 3000)  # Ici, on choisit un nombre fixe de nœuds aléatoires.
    seed_nodes = hub_seeds + seed_nodes1  # La combinaison des deux formes les nœuds initiaux.

    # Échantillonnage par marche aléatoire
    G_sampled = directed_random_walk_sampling(G, seed_nodes, walk_length=walk_length, sample_size=sample_size, p_restart=p_restart)
    
    # Paramètres ajustables :
    # - Une longueur de marche plus grande produit un graphe plus global.
    # - Une probabilité de redémarrage élevée permet d'éviter les chemins terminés ou les nœuds avec de nombreux arcs sortants.

    print(f"Le sous-graphe échantillonné contient {G_sampled.number_of_nodes()} nœuds et {G_sampled.number_of_edges()} arêtes.")
    print(f"Le graphe original contient {G.number_of_nodes()} nœuds et {G.number_of_edges()} arêtes.")

    # Extraction des arêtes avec leurs relations
    edges = [(u, v, d["relation"]) for u, v, d in G_sampled.edges(data=True)]

    # Sauvegarde des triplets dans un fichier texte
    save_triplets_to_txt(edges, file_path=f'{output_folder}/KG_train.txt')

    # Retourne le sous-graphe échantillonné
    return G_sampled
