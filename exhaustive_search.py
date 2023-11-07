import itertools
import networkx as nx
import matplotlib.pyplot as plt

def exhaustive_search(graph, k):
    edges = list(graph.edges)
    for r in range(1, len(edges) + 1):
        for subset in itertools.combinations(edges, r):
            if is_edge_dominating_set(graph, subset):
                if len(subset) == k:
                    return subset
    return None

def greedy_heuristic(graph, k):
    edge_set = set()
    sorted_edges = sorted(graph.edges, key=lambda e: graph.degree(e[0]) + graph.degree(e[1]))
    for edge in sorted_edges:
        if not any(is_adjacent(edge, e) for e in edge_set):
            edge_set.add(edge)
            if len(edge_set) == k:
                return edge_set
    return None

def is_edge_dominating_set(graph, edge_set):
    for edge in graph.edges:
        if not any(is_adjacent(edge, e) for e in edge_set):
            return False
    return True

def is_adjacent(edge1, edge2):
    return len(set(edge1) & set(edge2)) > 0

def main():
    #initialize graph:
    edges = 5
    vertices = 5    #vertices or nodes
    seed = 103453   #my nmec number
    G = nx.gnm_random_graph(vertices, edges, seed=seed)
    nx.draw(G=G)
    plt.show()

if __name__=="__main__":
    main()


