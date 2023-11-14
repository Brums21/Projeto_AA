import itertools
import networkx as nx
import networkx.algorithms.approximation as alg
import matplotlib.pyplot as plt
import time
import random
import math

random.seed(103453)

grd_basic_operations = 0
grd_tested_configurations = 0
exh_basic_operations = 0
exh_tested_configurations = 0

def exhaustive_search(graph, k):
    global exh_basic_operations
    global exh_tested_configurations
    exh_basic_operations = 0
    exh_tested_configurations = 0
    solutions = []
    edges = list(graph.edges)
    
    for subset in itertools.combinations(edges, k):
        exh_tested_configurations += 1
        is_dominating = True
        for edge in graph.edges:
            exh_basic_operations += 1
            if not any(is_adjacent(edge, e) for e in subset):
                is_dominating = False
                break
        if is_dominating:
            solutions.append(subset)
    
    return solutions, exh_basic_operations, exh_tested_configurations

def is_adjacent(edge1, edge2):
    return len(set(edge1) & set(edge2)) > 0

def greedy_heuristic(G, k):
    global grd_basic_operations
    global grd_tested_configurations
    grd_basic_operations = 0
    grd_tested_configurations = 0

    dominating_set = set()
    covered_edges = set()

    while len(dominating_set) < k:
        grd_tested_configurations += 1

        edge_candidates = []

        for edge in G.edges():
            grd_basic_operations += 1
            if edge not in dominating_set:
                adjacent_edges = set(G.edges(edge[0])) | set(G.edges(edge[1]))
                new_coverage = len(adjacent_edges - covered_edges)
                remaining = k - len(dominating_set)
                efficiency = min(new_coverage, remaining) / len(adjacent_edges)
                edge_candidates.append((edge, efficiency))

        grd_basic_operations += 1
        if not edge_candidates:
            break

        edge_candidates.sort(key=lambda x: x[1], reverse=True)
        best_candidate = edge_candidates[0][0]

        dominating_set.add(best_candidate)
        covered_edges.update(G.edges(best_candidate[0]))
        covered_edges.update(G.edges(best_candidate[1]))
        grd_basic_operations += 3

    return list(dominating_set) if len(dominating_set) <= k else None, grd_basic_operations, grd_tested_configurations

def draw_edges_and_graph(G, dominating_set):
    if dominating_set != None:
        for e in G.edges():
            G[e[0]][e[1]]['color'] = '#2ebcd0'
        for element in dominating_set:
            G[element[0]][element[1]]['color'] = 'red'
    edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
    pos = nx.get_node_attributes(G, 'pos')
    #image_dominating_set(G, edge_color_list, pos)
        
def image_dominating_set(G, edge_color_list, pos):
    plt.title("Edge dominating set of a graph with " +  str(len(G.edges()))+ " edges and " + str(len(G.nodes())) + " nodes")
    nx.draw(G, pos, node_color='#5dc66d', edge_color = edge_color_list, with_labels = True)
    directory = "edge_dominating_set_example.png"
    plt.savefig(str("images/" + directory))

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

def write_to_file(node_number, percentage_k, edges, to):
    text = "./info_nodes/" + to + ".txt"
    f = open(text, "a")
    f.write("nnodes:" + str(node_number) + "\n")
    f.write("percentage: " + str(percentage_k) + "\n")
    f.write("edges: " + str(edges) + "\n")
    f.close()

def write_graph_nsolutions_ex(solutions, k_value):
    f = open("./info_nodes/nsolutions_ex.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("nsolutions:" + str(solutions) + "\n")
    f.close()

def write_graph_exec_time_ex(execution_time, k_value):
    f = open("./info_nodes/exec_time_ex.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("execution_time:" + str(execution_time) + "\n")
    f.close()

def write_graph_basic_ex(basic, k_value):
    f = open("./info_nodes/basic_ex.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("basic_operations:" + str(basic) + "\n")
    f.close()

def write_graph_configurations_ex(configurations, k_value):
    f = open("./info_nodes/configurations_ex.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("configurations:" + str(configurations) + "\n")
    f.close() 

def write_graph_nsolutions_gr(solutions, k_value):
    f = open("./info_nodes/nsolutions_gr.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("nsolutions:" + str(solutions) + "\n")
    f.close()

def write_graph_exec_time_gr(execution_time, k_value):
    f = open("./info_nodes/exec_time_gr.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("execution_time:" + str(execution_time) + "\n")
    f.close()

def write_graph_basic_gr(basic, k_value):
    f = open("./info_nodes/basic_gr.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("basic_operations:" + str(basic) + "\n")
    f.close()

def write_graph_configurations_gr(configurations, k_value):
    f = open("./info_nodes/configurations_gr.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("configurations:" + str(configurations) + "\n")
    f.close() 

def write_graph_solutions_ex(solutions, k_value):
    f = open("./info_nodes/solutions_ex.txt", "a")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("solutions:" + str(solutions) + "\n")
    f.close() 
    pass

def write_graph_solutions_gr(solutions, k_value):
    f = open("./info_nodes/solutions_gr.txt", "a")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("solutions:" + str(solutions) + "\n")
    f.close() 
    pass

def initialize_files():
    f = open("./info_nodes/nsolutions_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/exec_time_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/basic_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/configurations_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/nsolutions_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/exec_time_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/basic_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/configurations_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/solutions_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/solutions_gr.txt", "w") #delete existing data
    f.close()

def write_nodes_edges_gr(nodes, percentage, nedges): 
    write_to_file(nodes, percentage, nedges, "nsolutions_gr")
    write_to_file(nodes, percentage, nedges, "exec_time_gr")
    write_to_file(nodes, percentage, nedges, "basic_gr")
    write_to_file(nodes, percentage, nedges, "configurations_gr")
    write_to_file(nodes, percentage, nedges, "solutions_gr")

def write_nodes_edges_ex(nodes, percentage, nedges):
    write_to_file(nodes, percentage, nedges, "nsolutions_ex")
    write_to_file(nodes, percentage, nedges, "exec_time_ex")
    write_to_file(nodes, percentage, nedges, "basic_ex")
    write_to_file(nodes, percentage, nedges, "configurations_ex")
    write_to_file(nodes, percentage, nedges, "solutions_ex")

def write_to_file_gr(nsolutions, solutions, execution_time, greedy_basic, greedy_configurations, kvalue):
    write_graph_nsolutions_gr(nsolutions, kvalue)
    write_graph_solutions_gr(solutions, kvalue)
    write_graph_exec_time_gr(execution_time, kvalue)
    write_graph_basic_gr(greedy_basic, kvalue)
    write_graph_configurations_gr(greedy_configurations, kvalue)

def write_to_file_ex(nsolutions, solutions_, execution_time, exh_basic, exh_configurations, kvalue):
    write_graph_nsolutions_ex(nsolutions, kvalue)
    write_graph_solutions_ex(solutions_, kvalue)
    write_graph_exec_time_ex(execution_time, kvalue)
    write_graph_basic_ex(exh_basic, kvalue)
    write_graph_configurations_ex(exh_configurations, kvalue)

def main():
    initialize_files()
    percentages = [0.125, 0.25, 0.5, 0.75]
    total = 0
    correct = 0

    for nodes in range(4, 30):
        for percentage in percentages:
            G, nedges = create_graph(nodes, percentage)
            k = [round(i*nedges) for i in percentages]
            write_nodes_edges_gr(nodes, percentage, nedges)

            if nodes<=8:
                write_nodes_edges_ex(nodes, percentage, nedges)
                pass

            for kvalue in k:
                start_time = time.time()
                solutions, greedy_basic, greedy_configurations = greedy_heuristic(G, kvalue)
                end_time = time.time()
                execution_time = end_time-start_time

                if solutions:
                    nsolutions = 1
                else:
                    nsolutions = 0

                write_to_file_gr(nsolutions, solutions, execution_time, greedy_basic, greedy_configurations, kvalue)

                #exaustive
                if nodes<=8:
                    start_time = time.time()
                    solutions_, exh_basic, exh_configurations = exhaustive_search(G, kvalue)
                    end_time = time.time()
                    execution_time = end_time-start_time
                    nsolutions = len(solutions_)
                    
                    write_to_file_ex(nsolutions, solutions_, execution_time, exh_basic, exh_configurations, kvalue)

                    if solutions_!=[] and percentage==0.75 and kvalue==8 and nodes==7:
                        draw_edges_and_graph(G, solutions_[0])

                    #check if solution is accurate
                    #solutions_ -> exhaustive solutions
                    #solutions -> greedy solutions
                    solutions_ = [list(solution) for solution in solutions_]
                    total+=1
                    correct = (correct+1) if is_solution_in_solutions(solutions, solutions_) else correct
                    
    correct_wrong(correct, total)

def is_solution_in_solutions(solution, solutions):
    #solution -> greedy
    #solutions -> exhaustive
    solution_set = set(solution)
    for s in solutions:
        if solution_set == set(s):
            return True
    if solutions == [] and solution_set == set():
        return True
    return False
                          
def plot_basic_op_gr_ex():

    with open("./info_nodes/basic_ex.txt", "r") as file:
        lines_ex = file.readlines()

    basic_op_k_05_1 = []
    basic_op_k_05_2 = []
    basic_op_k_05_3 = []
    basic_op_k_05_4 = []

    for i in range(0, len(lines_ex), 60):
        # Parse relevant information from each block of data
        basic_op_k_05_1.append(int(lines_ex[i + 11].split(":")[1].strip()))
        basic_op_k_05_2.append(int(lines_ex[i + 26].split(":")[1].strip()))
        basic_op_k_05_3.append(int(lines_ex[i + 41].split(":")[1].strip()))
        basic_op_k_05_4.append(int(lines_ex[i + 56].split(":")[1].strip()))

    plt.figure(figsize=(10, 6))
    nnodes = [i for i in range(4, 9)]

    plt.plot(nnodes, basic_op_k_05_1, label='k = 12.5%')
    plt.plot(nnodes, basic_op_k_05_2, label='k = 25%')
    plt.plot(nnodes, basic_op_k_05_3, label='k = 50%')
    plt.plot(nnodes, basic_op_k_05_4, label='k = 75%')
    plt.title('Basic exhaustive operations for different k values')
    plt.ylabel('Basic Operations')
    plt.xlabel('Number of nodes')
    plt.legend()

    plt.savefig("./images/basic_op_exhaustive_k_values.png")

    with open("./info_nodes/basic_gr.txt", "r") as file:
        lines_gr = file.readlines()

    basic_op_k_05_1 = [] #all these should, in the end, have len == 60
    basic_op_k_05_2 = []
    basic_op_k_05_3 = []
    basic_op_k_05_4 = []

    for i in range(0, len(lines_gr), 60):
        # Parse relevant information from each block of data
        basic_op_k_05_1.append(int(lines_gr[i + 11].split(":")[1].strip()))
        basic_op_k_05_2.append(int(lines_gr[i + 26].split(":")[1].strip()))
        basic_op_k_05_3.append(int(lines_gr[i + 41].split(":")[1].strip()))
        basic_op_k_05_4.append(int(lines_gr[i + 56].split(":")[1].strip()))

    plt.figure(figsize=(10, 6))
    nnodes = [i for i in range(4, 30)]

    print(nnodes)
    print(basic_op_k_05_1)

    plt.plot(nnodes, basic_op_k_05_1, label='k = 12.5%')
    plt.plot(nnodes, basic_op_k_05_2, label='k = 25%')
    plt.plot(nnodes, basic_op_k_05_3, label='k = 50%')
    plt.plot(nnodes, basic_op_k_05_4, label='k = 75%')
    plt.title('Basic greedy operations for different k values')
    plt.ylabel('Basic Operations')
    plt.xlabel('Number of nodes')
    plt.legend()

    plt.savefig("./images/basic_op_greedy_k_values.png")

def plot_exec_time_gr_ex():
    with open("./info_nodes/exec_time_ex.txt", "r") as file:
        lines_ex = file.readlines()

    exec_time_k_05_1 = []
    exec_time_k_05_2 = []
    exec_time_k_05_3 = []
    exec_time_k_05_4 = []

    for i in range(0, len(lines_ex), 60):
        # Parse relevant information from each block of data
        exec_time_k_05_1.append(float(lines_ex[i + 14].split(":")[1].strip()))
        exec_time_k_05_2.append(float(lines_ex[i + 29].split(":")[1].strip()))
        exec_time_k_05_3.append(float(lines_ex[i + 44].split(":")[1].strip()))
        exec_time_k_05_4.append(float(lines_ex[i + 59].split(":")[1].strip()))

    plt.figure(figsize=(10, 6))
    nnodes = [i for i in range(4, 9)]

    plt.plot(nnodes, exec_time_k_05_1, label='k = 12.5%')
    plt.plot(nnodes, exec_time_k_05_2, label='k = 25%')
    plt.plot(nnodes, exec_time_k_05_3, label='k = 50%')
    plt.plot(nnodes, exec_time_k_05_4, label='k = 75%')
    plt.title('Execution time for exhaustive algorithm for different k values')
    plt.ylabel('Execution time')
    plt.xlabel('Number of nodes')
    plt.legend()

    plt.savefig("./images/time_execution_exhaustive_k_values.png")
    
    with open("./info_nodes/exec_time_gr.txt", "r") as file:
        lines_gr = file.readlines()

    exec_time_k_05_1 = [] #all these should, in the end, have len == 60
    exec_time_k_05_2 = []
    exec_time_k_05_3 = []
    exec_time_k_05_4 = []

    for i in range(0, len(lines_gr), 60):
        # Parse relevant information from each block of data
        exec_time_k_05_1.append(float(lines_gr[i + 14].split(":")[1].strip()))
        exec_time_k_05_2.append(float(lines_gr[i + 29].split(":")[1].strip()))
        exec_time_k_05_3.append(float(lines_gr[i + 44].split(":")[1].strip()))
        exec_time_k_05_4.append(float(lines_gr[i + 59].split(":")[1].strip()))

    plt.figure(figsize=(10, 6))
    nnodes = [i for i in range(4, 30)]

    plt.plot(nnodes, exec_time_k_05_1, label='k = 12.5%')
    plt.plot(nnodes, exec_time_k_05_2, label='k = 25%')
    plt.plot(nnodes, exec_time_k_05_3, label='k = 50%')
    plt.plot(nnodes, exec_time_k_05_4, label='k = 75%')
    plt.title('Execution time for greedy algorithm for different k values')
    plt.ylabel('Execution time')
    plt.xlabel('Number of nodes')
    plt.legend()

    plt.savefig("./images/time_execution_greedy_k_values.png")

def plot_numb_solution_ex():
    with open("./info_nodes/nsolutions_ex.txt", "r") as file:
        lines_ex = file.readlines()

    soltution_k_05_1 = []
    soltution_k_05_2 = []
    soltution_k_05_3 = []
    soltution_k_05_4 = []

    for i in range(0, len(lines_ex), 60):
        # Parse relevant information from each block of data
        soltution_k_05_1.append(float(lines_ex[i + 11].split(":")[1].strip()))
        soltution_k_05_2.append(float(lines_ex[i + 26].split(":")[1].strip()))
        soltution_k_05_3.append(float(lines_ex[i + 41].split(":")[1].strip()))
        soltution_k_05_4.append(float(lines_ex[i + 56].split(":")[1].strip()))

    plt.figure(figsize=(10, 6))
    nnodes = [i for i in range(4, 9)]

    plt.plot(nnodes, soltution_k_05_1, label='k = 12.5%')
    plt.plot(nnodes, soltution_k_05_2, label='k = 25%')
    plt.plot(nnodes, soltution_k_05_3, label='k = 50%')
    plt.plot(nnodes, soltution_k_05_4, label='k = 75%')
    plt.title('Number of solutions for exhaustive algorithm for different k values')
    plt.ylabel('Number of solutions')
    plt.xlabel('Number of nodes')
    plt.legend()

    plt.savefig("./images/numb_solutions_ex_k_values.png")
    
def plot_rel_time_operations():
    with open("./info_nodes/basic_gr.txt", "r") as file:
        lines_gr = file.readlines()

    basic_op_k_05_1 = [] #all these should, in the end, have len == 60
    basic_op_k_05_2 = []
    basic_op_k_05_3 = []
    basic_op_k_05_4 = []

    for i in range(0, len(lines_gr), 60):
        # Parse relevant information from each block of data
        basic_op_k_05_1.append(int(lines_gr[i + 11].split(":")[1].strip()))
        basic_op_k_05_2.append(int(lines_gr[i + 26].split(":")[1].strip()))
        basic_op_k_05_3.append(int(lines_gr[i + 41].split(":")[1].strip()))
        basic_op_k_05_4.append(int(lines_gr[i + 56].split(":")[1].strip()))

    with open("./info_nodes/exec_time_gr.txt", "r") as file:
        lines_gr = file.readlines()

    exec_time_k_05_1 = [] #all these should, in the end, have len == 60
    exec_time_k_05_2 = []
    exec_time_k_05_3 = []
    exec_time_k_05_4 = []

    for i in range(0, len(lines_gr), 60):
        exec_time_k_05_1.append(float(lines_gr[i + 14].split(":")[1].strip()))
        exec_time_k_05_2.append(float(lines_gr[i + 29].split(":")[1].strip()))
        exec_time_k_05_3.append(float(lines_gr[i + 44].split(":")[1].strip()))
        exec_time_k_05_4.append(float(lines_gr[i + 59].split(":")[1].strip()))

    _, hor = plt.subplots(4, 1, figsize=(8,12))

    hor[0].plot(exec_time_k_05_1, basic_op_k_05_1)
    hor[0].set_title('Relation between executed time and basic operations with k = 12.5%')

    hor[1].plot(exec_time_k_05_2, basic_op_k_05_2)
    hor[1].set_title('Relation between executed time and basic operations with k = 25%')

    hor[2].plot(exec_time_k_05_3, basic_op_k_05_3)
    hor[2].set_title('Relation between executed time and basic operations with k = 50%')

    hor[3].plot(exec_time_k_05_4, basic_op_k_05_2)
    hor[3].set_title('Relation between executed time and basic operations with k = 75%')

    for element in hor:
        element.set(xlabel='Executed time', ylabel='Number of basic operations')

    plt.tight_layout()

    plt.savefig("./images/relations.png")
    pass

def correct_wrong(correct, total):
    categories = ['Correct', 'Incorrect']
    values = [correct, total - correct]

    plt.bar(categories, values, color=['green', 'red'], width=0.3, align='center')
    plt.title('Greedy solutions comparing to exhaustive solutions')
    plt.ylabel('Count')
    plt.savefig("./images/correct_solutions.png")

def print_plots():
    #plot 1 - nopertaions / number of nodes - for a given k - greedy vs heuristic
    plot_basic_op_gr_ex()
    
    #plot 2 - timeconsumed / number of nodes - for a given k - greedy vs heuristic
    plot_exec_time_gr_ex()

    #plot 3 - numb solutions / number of nodes - for a given k - greedy vs heuristic
    plot_numb_solution_ex()

    #plot 4 - 4 graphs with the relation between basic operations and time taken to perform
    plot_rel_time_operations()

    pass

if __name__=="__main__":
    main()
    print_plots()
    