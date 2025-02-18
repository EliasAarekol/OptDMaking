import networkx as nx
import matplotlib.pyplot as plt

def build_graph(graph, node):
    if node is None:
        return
    
    graph.add_node(node, bounds=node.bounds)
    if node.children:
        for child in node.children:
            if child:
                graph.add_edge(node, child)
                build_graph(graph, child)


def visualize_tree(root):
    graph = nx.DiGraph()
    build_graph(graph, root)
    
    pos = nx.nx_agraph.graphviz_layout(graph, prog="dot")  # Hierarchical layout
    labels = {node: node.bounds for node in graph.nodes}  # Display value and bounds
    
    node_colors = []
    for node in graph.nodes:
        if node == root:
            node_colors.append("red")  # Root node color
        elif not list(graph.successors(node)):
            node_colors.append("green")  # Leaf nodes color
        else:
            node_colors.append("lightblue")  # Intermediate nodes color
    
    plt.figure(figsize=(8, 6))
    nx.draw(graph, pos, with_labels=True, labels=labels, node_color=node_colors, edge_color='gray', node_size=2000, font_size=8, bbox=dict(facecolor="white", edgecolor="black", boxstyle="round,pad=0.3"))
    plt.show()
