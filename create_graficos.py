from ProjetoAA import random, nx, math

def create_graph(nvertices, percentage):  
    #para um grafo simples, o numero maximo de edges é nC2, n é o numero de vertices
    max_edges = round(nvertices * (nvertices-1) / 2)
    edges = int(max_edges*percentage)
    
    G = nx.Graph()

    #add vertices/nodes
    vertices_pos = []
    for i in range(nvertices):
        vertice_pos = create_vertice(vertices_pos, coordinates=[1,100])
        vertices_pos.append(vertice_pos)
        G.add_node(i, pos=vertice_pos)

    #add edges
    possible_edges = [(u, v) for u in G.nodes() for v in G.nodes() if u < v]
    random.shuffle(possible_edges)
    edges_to_add = possible_edges[:edges]
    G.add_edges_from(edges_to_add)
        
    return G, len(edges_to_add)

def create_vertice(vertices, coordinates):
    while True:
        x = random.randint(coordinates[0], coordinates[1])
        y = random.randint(coordinates[0], coordinates[1])
        if all(math.sqrt((x-vx)**2 + (y-vy)**2)>=9 for vx, vy in vertices):
            return (x, y)