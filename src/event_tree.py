import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout

f = open('event_tree.txt')
line_list = []
for line in f:
    text = line.strip('\n')
    if len(text) > 0:
        item = eval(text)
        line_list.append(item)
parallel = line_list[0]
leaves = line_list[1:]
tree = {}
depth = 0
T = nx.DiGraph()
for leave in leaves:
    code = leave[1]
    if depth < len(code):
        depth = len(code)
    tree[code] = [leave[0],1]
    l = len(code)
    for i in range(l-1,-1,-1):
        parent_code = code[:i]
        child_code = code[:i+1]
        if parent_code not in tree:
            if parent_code in parallel:
                tree[parent_code] = ['p',1]
            else:
                tree[parent_code] = ['s',1]
            T.add_edge(parent_code, child_code)
        else:
            T.add_edge(parent_code, child_code)
            break

for node in T:
    if T.out_degree(node) > 0 and tree[node][0] == 's':
        tree[node][1] = len([n for n in T.neighbors(node)])

print(tree)
nx.draw(T, with_labels=True)
plt.show()