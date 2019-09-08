import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


def initialize_event_tree(leave_file):
    f = open(leave_file)
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
    t = nx.DiGraph()
    for leave in leaves:
        code = leave[1]
        if depth < len(code):
            depth = len(code)
        tree[code] = [leave[0], 1]
        for i in range(len(code)-1, -1, -1):
            parent_code = code[:i]
            child_code = code[:i+1]
            if parent_code not in tree:
                if parent_code in parallel:
                    tree[parent_code] = ['p', 1]
                else:
                    tree[parent_code] = ['s', 1]
                t.add_edge(parent_code, child_code)
            else:
                t.add_edge(parent_code, child_code)
                break

    for node in t:
        num = t.out_degree(node)
        if num > 0 and tree[node][0] == 's':
            tree[node][1] = len([n for n in t.neighbors(node)])

    return tree, t


def main():
    leave_file = 'event_tree.txt'
    tree, t = initialize_event_tree(leave_file)
    print(tree)
    pos = graphviz_layout(t, prog='dot')
    nx.draw(t, pos, arrows=False, with_labels=False)
    plt.show()


main()