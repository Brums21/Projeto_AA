import itertools
import time
import random
import print_graficos as pg
import write_graficos as wg
import create_graficos as cg
import create_sw_graficos as cswg

random.seed(103453)

random_operations = 0
random_configurations = 0
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
            solutions.append(list(subset))
    
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


def random_algorithm(graph, k, num_configurations):
    global random_operations
    global random_configurations
    
    random_operations = 0
    random_configurations = 0

    solution = None
    tested_solutions = set()

    start_time = time.time()

    for _ in range(num_configurations):
        random_operations +=1
        candidate = random.sample(list(graph.edges()), k)
        if str(candidate) in tested_solutions:
            random_configurations+=1
            continue
        tested_solutions.add(str(candidate))
        if is_valid_solution(graph, candidate):
            solution = candidate
            return solution, time.time()-start_time, random_operations, random_configurations

    return solution, time.time()-start_time, random_operations, random_configurations


def is_valid_solution(graph, candidate):
    global random_operations
    global random_configurations

    random_configurations+=1
    
    all_edges = set(graph.edges())

    covered_vertices = set()

    for edge in candidate:
        random_operations +=1
        covered_vertices.update(edge)

    for edge in all_edges:
        random_operations +=1
        if not set(edge).intersection(covered_vertices):
            return False
        
    return True


def is_solution_in_solutions(solution, solutions):
    solution_set = set(solution)
    for s in solutions:
        if solution_set == set(s):
            return True
    if solutions == [] and solution_set == set():
        return True
    return False

def main():
    wg.initialize_files()
    percentages = [0.125, 0.25, 0.5, 0.75]
    total = 0

    # exhaustive:
    for nodes in range(4, 110):
        for percentage in percentages:
            G, nedges = cg.create_graph(nodes, percentage)
            k = [round(i*nedges) for i in percentages]
            
            if nodes<=8:
                wg.write_nodes_edges_ex(nodes, percentage, nedges)
            if nodes<=30:
                wg.write_nodes_edges_gr(nodes, percentage, nedges)
            if nodes<=110:
                wg.write_nodes_edges_rnd(nodes, percentage, nedges)
                
            for kvalue in k:

                if nodes<=8:
                    start_time = time.time()
                    solutions_ex, exh_basic, exh_configurations = exhaustive_search(G, kvalue)
                    end_time = time.time()
                    execution_time = end_time-start_time
                    nsolutions = len(solutions_ex)
                    
                    wg.write_to_file_ex(nsolutions, solutions_ex, execution_time, exh_basic, exh_configurations, kvalue)

                    # draw one exemaple graph
                    if solutions_ex!=[] and percentage==0.75 and kvalue==8 and nodes==7:
                        pg.draw_edges_and_graph(G, solutions_ex[0])

                    solutions_ex = [list(solution) for solution in solutions_ex]
                    total+=1

                # greedy
                if nodes<=30:
                    start_time = time.time()
                    solutions, greedy_basic, greedy_configurations = greedy_heuristic(G, kvalue)
                    end_time = time.time()
                    execution_time = end_time-start_time

                    if solutions:
                        nsolutions = 1
                    else:
                        nsolutions = 0

                    wg.write_to_file_gr(nsolutions, solutions, execution_time, greedy_basic, greedy_configurations, kvalue)

                # random:
                if nodes<=110:
                    solution_random, random_exec_time, numb_rand_basic, numb_rand_config = random_algorithm(G, kvalue, 10)

                    if solution_random:
                        nsolutions = 1
                    else:
                        nsolutions = 0

                    wg.write_to_file_random_sw(nsolutions, solution_random, random_exec_time, numb_rand_basic, numb_rand_config, kvalue)


    #for wg graphs:
    
    G_small = cswg.create_small_graph()
    G_medium = cswg.create_medium_graph()
    G_large = cswg.create_large_graph()

    graphs = [G_small, G_medium, G_large]

    for graph in graphs:
        wg.write_nodes_edges_rnd(graph.number_of_nodes(), "100%", graph.number_of_edges())
        k = [round(i*graph.number_of_edges()) for i in percentages]
        print("graph:", graph)
        for kvalue in k:
            solution_random, random_exec_time, numb_rand_basic, numb_rand_config = random_algorithm(graph, kvalue, 3)
            
            if solution_random:
                nsolutions = 1
            else:
                nsolutions = 0
                    
            wg.write_to_file_random_sw(nsolutions, solution_random, random_exec_time, numb_rand_basic, numb_rand_config, kvalue)


if __name__=="__main__":
    #main()
    pg.print_plots()
    