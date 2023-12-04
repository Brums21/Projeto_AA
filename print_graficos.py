from ProjetoAA import plt, nx

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

def correct_wrong(correct, total):
    categories = ['Correct', 'Incorrect']
    values = [correct, total - correct]

    plt.bar(categories, values, color=['green', 'red'], width=0.3, align='center')
    plt.title('Greedy solutions comparing to exhaustive solutions')
    plt.ylabel('Count')
    plt.savefig("./images/correct_solutions.png")

def draw_edges_and_graph(G, dominating_set):
    if dominating_set != None:
        for e in G.edges():
            G[e[0]][e[1]]['color'] = '#2ebcd0'
        for element in dominating_set:
            G[element[0]][element[1]]['color'] = 'red'
    edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
    pos = nx.get_node_attributes(G, 'pos')
    image_dominating_set(G, edge_color_list, pos)
        
def image_dominating_set(G, edge_color_list, pos):
    plt.title("Edge dominating set of a graph with " +  str(len(G.edges()))+ " edges and " + str(len(G.nodes())) + " nodes")
    nx.draw(G, pos, node_color='#5dc66d', edge_color = edge_color_list, with_labels = True)
    directory = "edge_dominating_set_example.png"
    plt.savefig(str("images/" + directory))

def print_plots():
    #plot 1 - nopertaions / number of nodes - for a given k - greedy vs heuristic
    plot_basic_op_gr_ex()
    
    #plot 2 - timeconsumed / number of nodes - for a given k - greedy vs heuristic
    plot_exec_time_gr_ex()

    #plot 3 - numb solutions / number of nodes - for a given k - greedy vs heuristic
    plot_numb_solution_ex()

    #plot 4 - 4 graphs with the relation between basic operations and time taken to perform
    plot_rel_time_operations()