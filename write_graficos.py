def write_to_file(node_number, percentage_k, edges, to):
    text = "./info_nodes/" + to + ".txt"
    f = open(text, "a")
    f.write("nnodes:" + str(node_number) + "\n")
    f.write("percentage: " + str(percentage_k) + "\n")
    f.write("edges: " + str(edges) + "\n")
    f.close()

def write_graph_exec_time(execution_time, k_value, type):
    if type == "ex":
        f = open("./info_nodes/exec_time_ex.txt", "a")
    elif type == "gr":
        f = open("./info_nodes/exec_time_gr.txt", "a")
    else:
        f = open("./info_nodes/exec_time_rnd.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("execution_time:" + str(execution_time) + "\n")
    f.close()

def write_graph_basic(basic, k_value, type):
    if type == "ex":
        f = open("./info_nodes/basic_ex.txt", "a")
    elif type == "gr":
        f = open("./info_nodes/basic_gr.txt", "a")
    else:
        f = open("./info_nodes/basic_rnd.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("basic_operations:" + str(basic) + "\n")
    f.close()

def write_graph_configurations(configurations, k_value, type):
    if type == "ex":
        f = open("./info_nodes/configurations_ex.txt", "a")
    elif type == "gr":
        f = open("./info_nodes/configurations_gr.txt", "a")
    else:
        f = open("./info_nodes/configurations_rnd.txt", "a")
    f.write("--\n")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("configurations:" + str(configurations) + "\n")
    f.close() 

def write_graph_nsolutions(nsolutions, k_value, type):
    if type == "ex":
        f = open("./info_nodes/nsolutions_ex.txt", "a")
    elif type == "gr":
        f = open("./info_nodes/nsolutions_gr.txt", "a")
    else:
        f = open("./info_nodes/nsolutions_rnd.txt", "a")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("nsolutions:" + str(nsolutions) + "\n")
    f.close() 

def write_graph_solutions(solutions, k_value, type):
    if type == "ex":
        f = open("./info_nodes/solutions_ex.txt", "a")
    elif type == "gr":
        f = open("./info_nodes/solutions_gr.txt", "a")
    else:
        f = open("./info_nodes/solutions_rnd.txt", "a")
    f.write("k_value:" + str(k_value) + "\n")
    f.write("solutions:" + str(solutions) + "\n")
    f.close()

def initialize_files():
    f = open("./info_nodes/nsolutions_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/nsolutions_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/nsolutions_rnd.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/exec_time_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/exec_time_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/exec_time_rnd.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/basic_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/basic_rnd.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/basic_rnd.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/configurations_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/configurations_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/configurations_rnd.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/solutions_ex.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/solutions_gr.txt", "w") #delete existing data
    f.close()
    f = open("./info_nodes/solutions_rnd.txt", "w") #delete existing data
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

def write_nodes_edges_rnd(nodes, percentage, nedges):
    write_to_file(nodes, percentage, nedges, "nsolutions_rnd")
    write_to_file(nodes, percentage, nedges, "exec_time_rnd")
    write_to_file(nodes, percentage, nedges, "basic_rnd")
    write_to_file(nodes, percentage, nedges, "configurations_rnd")
    write_to_file(nodes, percentage, nedges, "solutions_rnd")

def write_to_file_gr(nsolutions, solutions, execution_time, greedy_basic, greedy_configurations, kvalue):
    write_graph_nsolutions(nsolutions, kvalue, "gr")
    write_graph_solutions(solutions, kvalue, "gr")
    write_graph_exec_time(execution_time, kvalue, "gr")
    write_graph_basic(greedy_basic, kvalue, "gr")
    write_graph_configurations(greedy_configurations, kvalue, "gr")

def write_to_file_ex(nsolutions, solutions_, execution_time, exh_basic, exh_configurations, kvalue):
    write_graph_nsolutions(nsolutions, kvalue, "ex")
    write_graph_solutions(solutions_, kvalue, "ex")
    write_graph_exec_time(execution_time, kvalue, "ex")
    write_graph_basic(exh_basic, kvalue, "ex")
    write_graph_configurations(exh_configurations, kvalue, "ex")

def write_to_file_random_sw(nsolutions, solutions_, execution_time, exh_basic, exh_configurations, kvalue):
    write_graph_nsolutions(nsolutions, kvalue, "rnd")
    write_graph_solutions(solutions_, kvalue, "rnd")
    write_graph_exec_time(execution_time, kvalue, "rnd")
    write_graph_basic(exh_basic, kvalue, "rnd")
    write_graph_configurations(exh_configurations, kvalue, "rnd")

    