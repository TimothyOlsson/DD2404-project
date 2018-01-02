# import networkx, install if necessary
try:
    import networkx as nx
except ImportError:
    import pip

    pip.main(['install', '--user', 'networkx'])
    import networkx as nx

G = nx.DiGraph()
pos = {}

for i in range(8):
    pos['X' + str(i)] = (i, 0)  # 8 neurons in the input layer
    pos['H' + str(i)] = (i, 1)  # 8 neurons in the first hidden layer

    for j in range(8): G.add_edge('X' + str(j), 'H' + str(i))

    if i < 4:
        pos['U' + str(i)] = (i + 2, 2)  # 4 neurons in the second hidden layer
        for j in range(8): G.add_edge('H' + str(j), 'U' + str(i))

    if i < 6:
        pos['Y' + str(i)] = (i + 1, 3)  # 6 neurons in the output layer
        for j in range(4): G.add_edge('U' + str(j), 'Y' + str(i))

nx.draw(G, pos, node_color='y', node_size=750)