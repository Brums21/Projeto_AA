from ProjetoAA import nx, plt

def create_large_graph():
    return create_graph_based_on_file("SWlargeG")

def create_medium_graph():
    return create_graph_based_on_file("SWmediumG")

def create_small_graph():
    return create_graph_based_on_file("SWtinyG")

def create_graph_based_on_file(filename):

    with open("./sw_grafos/" + filename + ".txt", "r") as file:
        grafic_lines = file.readlines()

    connections = []

    for i in range(4, len(grafic_lines)):
        line = grafic_lines[i].split()

        #skip gralhas
        if line[0] == line[1]:
            continue

        connection = tuple(int(x) for x in line)
        connections.append(connection)
        
    G = nx.Graph()

    G.add_edges_from(connections)

    return G

    