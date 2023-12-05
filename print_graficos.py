import networkx as nx
import matplotlib.pyplot as plt
import ast

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
    plt.close()

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

    plt.plot(nnodes, basic_op_k_05_1, label='k = 12.5%')
    plt.plot(nnodes, basic_op_k_05_2, label='k = 25%')
    plt.plot(nnodes, basic_op_k_05_3, label='k = 50%')
    plt.plot(nnodes, basic_op_k_05_4, label='k = 75%')
    plt.title('Basic greedy operations for different k values')
    plt.ylabel('Basic Operations')
    plt.xlabel('Number of nodes')
    plt.legend()

    plt.savefig("./images/basic_op_greedy_k_values.png")
    plt.close()

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
    plt.close()
    
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
    plt.close()

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
    plt.close()
    
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
    plt.close()

def plot_correct_wrong_ex_rnd():
    categories = ['Correct', 'Incorrect']

    with open("./info_nodes/solutions_rnd.txt", "r") as file:
        lines_rnd = file.readlines()
    
    with open("./info_nodes/solutions_ex.txt", "r") as file:
        lines_ex = file.readlines()

    solutions_ex = []
    solutions_rnd = []

    for i in range(0, len(lines_ex), 15):
        solutions_ex.append(lines_ex[i + 5].split(":")[1].strip()[1:-1])
        solutions_ex.append(lines_ex[i + 8].split(":")[1].strip()[1:-1])
        solutions_ex.append(lines_ex[i + 11].split(":")[1].strip()[1:-1])
        solutions_ex.append(lines_ex[i + 14].split(":")[1].strip()[1:-1])

        solutions_rnd.append(lines_rnd[i + 5].split(":")[1].strip())
        solutions_rnd.append(lines_rnd[i + 8].split(":")[1].strip())
        solutions_rnd.append(lines_rnd[i + 11].split(":")[1].strip())
        solutions_rnd.append(lines_rnd[i + 14].split(":")[1].strip())

    correct = 0

    for i in range(len(solutions_ex)):
        if solutions_ex[i] == '':
            solutions_ex[i] = '[]'

        solutions_ex_ = ast.literal_eval(solutions_ex[i])

        if type(solutions_ex_) is tuple:
            solutions_ex_ = list(solutions_ex_)

        if len(solutions_ex_)>0 and type(solutions_ex_[0]) is tuple:
            solutions_ex_ = [solutions_ex_]

        if solutions_rnd[i] == "None":
            solutions_rnd[i] = '[]'

        random = tuple(eval(solutions_rnd[i]))

        if (solutions_ex_ == [] and set(random) == set()):
            correct+=1

        for element in solutions_ex_:
            el = tuple(element)
            if set(random) == set(el):
                correct+=1
    
    incorrect = len(solutions_ex) - correct

    values = [correct, incorrect]
    plt.bar(categories, values, color=['green', 'red'], width=0.3, align='center')
    plt.title('Randomized solutions comparing to exhaustive solutions')
    plt.ylabel('Count')
    plt.savefig("./images/correct_solutions_rnd.png")
    plt.close()
    
def plot_correct_wrong_ex_gr():
    categories = ['Correct', 'Incorrect']

    with open("./info_nodes/solutions_gr.txt", "r") as file:
        lines_gr = file.readlines()
    
    with open("./info_nodes/solutions_ex.txt", "r") as file:
        lines_ex = file.readlines()

    solutions_ex = []
    solutions_gr = []

    for i in range(0, len(lines_ex), 15):
        solutions_ex.append(lines_ex[i + 5].split(":")[1].strip()[1:-1])
        solutions_ex.append(lines_ex[i + 8].split(":")[1].strip()[1:-1])
        solutions_ex.append(lines_ex[i + 11].split(":")[1].strip()[1:-1])
        solutions_ex.append(lines_ex[i + 14].split(":")[1].strip()[1:-1])

        solutions_gr.append(lines_gr[i + 5].split(":")[1].strip())
        solutions_gr.append(lines_gr[i + 8].split(":")[1].strip())
        solutions_gr.append(lines_gr[i + 11].split(":")[1].strip())
        solutions_gr.append(lines_gr[i + 14].split(":")[1].strip())

    correct = 0

    for i in range(len(solutions_ex)):
        if solutions_ex[i] == '':
            solutions_ex[i] = '[]'

        solutions_ex_ = ast.literal_eval(solutions_ex[i])

        if type(solutions_ex_) is tuple:
            solutions_ex_ = list(solutions_ex_)

        if len(solutions_ex_)>0 and type(solutions_ex_[0]) is tuple:
            solutions_ex_ = [solutions_ex_]

        if solutions_gr[i] == "None":
            solutions_gr[i] = '[]'

        greedy = tuple(eval(solutions_gr[i]))

        if solutions_ex_ == [] and set(greedy) == set():
            correct+=1

        for element in solutions_ex_:
            el = tuple(element)
            if set(greedy) == set(el):
                correct+=1

    incorrect = len(solutions_ex) - correct
    values = [correct, incorrect]
    plt.bar(categories, values, color=['green', 'red'], width=0.3, align='center')
    plt.title('Greedy solutions comparing to exhaustive solutions')
    plt.ylabel('Count')
    plt.savefig("./images/correct_solutions_gr.png")
    plt.close()

def plot_scatter_random_time_consumed():
    with open("./info_nodes/exec_time_rnd.txt", "r") as file:
        lines_rand = file.readlines()

    nnodes = []
    time_125 = []
    time_250 = []
    time_50 = []
    time_75 = []

    for i in range(0, len(lines_rand)-45, 60):
        nnodes.append(int(lines_rand[i].split(":")[1].strip()))
        time_125.append(float(lines_rand[i + 14].split(":")[1].strip()))
        time_250.append(float(lines_rand[i + 29].split(":")[1].strip()))
        time_50.append(float(lines_rand[i + 44].split(":")[1].strip()))
        time_75.append(float(lines_rand[i + 59].split(":")[1].strip()))


    plt.scatter(nnodes, time_125, marker='x', color=(0.678, 0.176, 0.153, 1), label='k=12.5%')
    plt.scatter(nnodes, time_250, marker='x', color=(0.404, 0.722, 0.431, 1), label='k=25%')
    plt.scatter(nnodes, time_50, marker='x', color=(0.169, 0.631, 0.831, 1), label='k=50%')
    plt.scatter(nnodes, time_75, marker='x', color=(0.922, 0.651, 0.216, 1), label='k=75%')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Number of nodes related to execution time for different edge percentages')
    plt.legend()

    plt.savefig("./images/plot_scatter_random_time_consumed.png")
    plt.close()

def plot_number_of_edges_solutions_found():
    with open("./info_nodes/nsolutions_rnd.txt", "r") as file:
        lines_rand = file.readlines()

    nnodes = []
    solutions_125_0 = []
    solutions_250_0 = []
    solutions_50_0 = []
    solutions_75_0 = []

    solutions_125_1 = []
    solutions_250_1 = []
    solutions_50_1 = []
    solutions_75_1 = []

    solutions_125_2 = []
    solutions_250_2 = []
    solutions_50_2 = []
    solutions_75_2 = []

    solutions_125_3 = []
    solutions_250_3 = []
    solutions_50_3 = []
    solutions_75_3 = []


    for i in range(0, len(lines_rand)-45, 60):
        nnodes.append(int(lines_rand[i].split(":")[1].strip()))

        solutions_125_0.append(int(lines_rand[i + 5].split(":")[1].strip()))
        solutions_250_0.append(int(lines_rand[i + 20].split(":")[1].strip()))
        solutions_50_0.append(int(lines_rand[i + 35].split(":")[1].strip()))
        solutions_75_0.append(int(lines_rand[i + 50].split(":")[1].strip()))

        solutions_125_1.append(int(lines_rand[i + 8].split(":")[1].strip()))
        solutions_250_1.append(int(lines_rand[i + 23].split(":")[1].strip()))
        solutions_50_1.append(int(lines_rand[i + 38].split(":")[1].strip()))
        solutions_75_1.append(int(lines_rand[i + 53].split(":")[1].strip()))

        solutions_125_2.append(int(lines_rand[i + 11].split(":")[1].strip()))
        solutions_250_2.append(int(lines_rand[i + 26].split(":")[1].strip()))
        solutions_50_2.append(int(lines_rand[i + 41].split(":")[1].strip()))
        solutions_75_2.append(int(lines_rand[i + 56].split(":")[1].strip()))

        solutions_125_3.append(int(lines_rand[i + 14].split(":")[1].strip()))
        solutions_250_3.append(int(lines_rand[i + 29].split(":")[1].strip()))
        solutions_50_3.append(int(lines_rand[i + 44].split(":")[1].strip()))
        solutions_75_3.append(int(lines_rand[i + 59].split(":")[1].strip()))

    
    _, hor = plt.subplots(4, 1, figsize=(10,14))
    plt.subplots_adjust(hspace=0.8)

    hor[0].scatter(nnodes, solutions_125_0, marker='x', color=(0.678, 0.176, 0.153, 1), label='k=12.5%')
    hor[0].scatter(nnodes, solutions_250_0, marker='x', color=(0.404, 0.722, 0.431, 1), label='k=25%')
    hor[0].scatter(nnodes, solutions_50_0, marker='x', color=(0.169, 0.631, 0.831, 1), label='k=50%')
    hor[0].scatter(nnodes, solutions_75_0, marker='x', color=(0.922, 0.651, 0.216, 1), label='k=75%')
    hor[0].set_title('Number of nodes related to the number of solutions with 12.5% number of edges')
    hor[0].legend()

    hor[1].scatter(nnodes, solutions_125_1, marker='x', color=(0.678, 0.176, 0.153, 1), label='k=12.5%')
    hor[1].scatter(nnodes, solutions_250_1, marker='x', color=(0.404, 0.722, 0.431, 1), label='k=25%')
    hor[1].scatter(nnodes, solutions_50_1, marker='x', color=(0.169, 0.631, 0.831, 1), label='k=50%')
    hor[1].scatter(nnodes, solutions_75_1, marker='x', color=(0.922, 0.651, 0.216, 1), label='k=75%')
    hor[1].set_title('Number of nodes related to the number of solutions with 25% number of edges')
    hor[1].legend()

    hor[2].scatter(nnodes, solutions_125_2, marker='x', color=(0.678, 0.176, 0.153, 1), label='k=12.5%')
    hor[2].scatter(nnodes, solutions_250_2, marker='x', color=(0.404, 0.722, 0.431, 1), label='k=25%')
    hor[2].scatter(nnodes, solutions_50_2, marker='x', color=(0.169, 0.631, 0.831, 1), label='k=50%')
    hor[2].scatter(nnodes, solutions_75_2, marker='x', color=(0.922, 0.651, 0.216, 1), label='k=75%')
    hor[2].set_title('Number of nodes related to the number of solutions with 50% number of edges')
    hor[2].legend()

    hor[3].scatter(nnodes, solutions_125_3, marker='x', color=(0.678, 0.176, 0.153, 1), label='k=12.5%')
    hor[3].scatter(nnodes, solutions_250_3, marker='x', color=(0.404, 0.722, 0.431, 1), label='k=25%')
    hor[3].scatter(nnodes, solutions_50_3, marker='x', color=(0.169, 0.631, 0.831, 1), label='k=50%')
    hor[3].scatter(nnodes, solutions_75_3, marker='x', color=(0.922, 0.651, 0.216, 1), label='k=75%')
    hor[3].set_title('Number of nodes related to the number of solutions with 75% number of edges')
    hor[3].legend()

    for element in hor:
        element.set(xlabel='Number of nodes', ylabel='Number of solutions')

    plt.savefig("./images/plot_number_of_edges_solutions_found.png")
    plt.close()

def plot_number_of_edges_noperations_found():
    with open("./info_nodes/basic_rnd.txt", "r") as file:
        lines_rand = file.readlines()

    nnodes = []
    basic_125 = []
    basic_250 = []
    basic_50 = []
    basic_75 = []

    for i in range(0, len(lines_rand)-45, 60):
        nnodes.append(int(lines_rand[i].split(":")[1].strip()))
        basic_125.append(int(lines_rand[i + 14].split(":")[1].strip()))
        basic_250.append(int(lines_rand[i + 29].split(":")[1].strip()))
        basic_50.append(int(lines_rand[i + 44].split(":")[1].strip()))
        basic_75.append(int(lines_rand[i + 59].split(":")[1].strip()))


    plt.plot(nnodes, basic_125, color=(0.678, 0.176, 0.153, 1), label='k=12.5%')
    plt.plot(nnodes, basic_250, color=(0.404, 0.722, 0.431, 1), label='k=25%')
    plt.plot(nnodes, basic_50, color=(0.169, 0.631, 0.831, 1), label='k=50%')
    plt.plot(nnodes, basic_75, color=(0.922, 0.651, 0.216, 1), label='k=75%')

    plt.xlabel('Number of Nodes')
    plt.ylabel('Number of operations')
    plt.title('Number of nodes related to number of operations - Randomized Algorithm', fontsize = 10)
    plt.legend()

    plt.savefig("./images/plot_line_random_basic_operations_rnd.png")
    plt.close()

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
    plt.close()


def print_plots():
    #plot 1 - nopertaions / number of nodes - for a given k - greedy vs heuristic
    #plot_basic_op_gr_ex()
    
    #plot 2 - timeconsumed / number of nodes - for a given k - greedy vs heuristic
    #plot_exec_time_gr_ex()

    #plot 3 - numb solutions / number of nodes - for a given k - greedy vs heuristic
    #plot_numb_solution_ex()

    #plot 4 - 4 graphs with the relation between basic operations and time taken to perform
    #plot_rel_time_operations()

    #plot 5 - correct vs wrong exhaustive comparing to greedy algorithm
    #plot_correct_wrong_ex_gr()

    #plot 6 - correct vs wrong exhaustive comparing to greedy algorithm
    #plot_correct_wrong_ex_rnd()

    #plot 7 - scatter plot random algorithm exec time
    #plot_scatter_random_time_consumed()

    #plot 8 - scatter plot random algorithm numb solutions found
    plot_number_of_edges_solutions_found()

    #plot 9 - line chart random algorithm numb basic operations
    #plot_number_of_edges_noperations_found()
    pass