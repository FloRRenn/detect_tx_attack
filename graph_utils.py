import json
import operator
import gmatch4py as gm
import networkx as nx
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

def build_graph_from_json(json_trace):
    sorted_trace = sorted(json_trace, key = operator.itemgetter(2, 5, 6))
    node_dict = {}
    graph = nx.DiGraph()

    for trace_entry in sorted_trace:
        if trace_entry[5] == "reference":
            continue
        
        source_node, target_node = trace_entry[3], trace_entry[4]

        if source_node not in node_dict:
            node_dict[source_node] = len(node_dict)
        if target_node not in node_dict:
            node_dict[target_node] = len(node_dict)

        source_node_id, target_node_id = node_dict[source_node], node_dict[target_node]

        graph.add_node(source_node_id, nodetype=trace_entry[0]['group'])
        graph.add_node(target_node_id, nodetype=trace_entry[1]['group'])
        graph.add_edge(source_node_id, target_node_id, edgetype = (trace_entry[5] + str(trace_entry[-1])))

    return graph

def visualize_graph(graph):
    pos = nx.spring_layout(graph)  # You can choose a different layout if needed
    node_colors = [graph.nodes[n]['nodetype'] for n in graph.nodes]
    edge_labels = {(u, v): d['edgetype'] for u, v, d in graph.edges(data=True)}

    nx.draw(graph, pos, with_labels=True, node_color=node_colors, cmap=plt.cm.Blues)
    nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels)

    plt.show()

def select_k(spectrum, minimum_energy = 0.9):
    running_total = 0.0
    total = sum(spectrum)
    if total == 0.0:
        return len(spectrum)
    
    for i in range(len(spectrum)):
        running_total += spectrum[i]
        if running_total / total >= minimum_energy:
            return i + 1
    return len(spectrum)

# Calculate Jaccard similarity coefficient between two sets.
def jaccard_sim(set_a, set_b):
    a = " ".join(list(set_a))
    b = " ".join(list(set_b))
    
    vectorizer = CountVectorizer(tokenizer = lambda s: s.split())
    tf_matrix = vectorizer.fit_transform([a, b]).toarray()
    
    intersection = np.sum(np.min(tf_matrix, axis = 0))
    union = np.sum(np.max(tf_matrix, axis = 0))
    sim_coeff = 1.0 * intersection / union if union != 0 else 0
    return sim_coeff

# Calculate the similarity between two graphs using the Graph Edit Distance (GED) metric
def calculate_graph_similarity_by_GED(graph_a, graph_b):
    similarity_calculator = gm.VertexEdgeOverlap()
    similarity_calculator.set_attr_graph_used("nodetype", "edgetype")
    
    similarity_matrix = similarity_calculator.compare([graph_a, graph_b], None)
    distances = similarity_calculator.distance(similarity_matrix)
    print("distance:", distances)
    # Calculate the average similarity between the two graphs
    average_similarity = (distances[0][1] + distances[1][0]) / 2.0
    return average_similarity

def calculate_graph_distance(trace_1, trace_2):
    try:
        trace_1 = json.loads(trace_1)
        trace_2 = json.loads(trace_2)
        
        graph1 = build_graph_from_json(trace_1)
        graph2 = build_graph_from_json(trace_2)
        
        # Calculate the similarity
        similarity_score = calculate_graph_similarity_by_GED(graph1, graph2)
        
        if np.isinf(similarity_score):
            similarity_score = 0
            if len(trace_1) == 2 or len(trace_2) == 2:
                similarity_score = abs(len(trace_1) - len(trace_2))     
        return similarity_score
    
    except Exception as e:
        print("Error")
        return 0