import networkx as nx

def build_graph(report):
    G = nx.DiGraph()

    for proc in report["behavior"]["processes"]:
        parent = proc["process_name"]
        for call in proc["calls"]:
            api = call["api"]
            G.add_edge(parent, api)

    return G