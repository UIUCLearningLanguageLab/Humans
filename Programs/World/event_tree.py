import networkx as nx
from World import config
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout


####################################################################################################################
# define event_tree generating function
# The structure of a tree is uniquely determined by the coordinate of all its leaves thus we generate the
# tree from the leave information in the leave file.
# There are two leave files in the directory, we use the 'event_tree' for experiment.
####################################################################################################################

def initialize_event_tree(leave_file, show_tree):
    f = open(leave_file)
    line_list = []
    for line in f:
        text = line.strip('\n')
        if len(text) > 0:
            item = eval(text)
            line_list.append(item)
    parallel = line_list[0] # the first line of leave file specify parallel nodes(parallel combination of sub events)
    prob_parallel = line_list[1]  # the 2nd line specify which of the parallel nodes choose child by random assignment
    leaves = line_list[2:]  # read the leaves
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
                    if parent_code in prob_parallel:  # generating tree dictionary which specify type of the node with
                        # initial state 1(unfinished)
                        tree[parent_code] = ['pp', 1]  # probabilistic parallel nodes
                    else:
                        tree[parent_code] = ['op', 1]  # parallel nodes determining child by computing scores
                else:
                    tree[parent_code] = ['s', 1]  # serial nodes
                t.add_edge(parent_code, child_code)
            else:
                t.add_edge(parent_code, child_code)
                break

    for node in t:
        num = t.out_degree(node)
        if num > 0 and tree[node][0] == 's':
            tree[node][1] = len([n for n in t.neighbors(node)]) # set specialized state for serial node, as the number
            # of its children

    if show_tree:
        pos = graphviz_layout(t, prog='dot')
        nx.draw(t, pos, arrows=False, with_labels=True)
        plt.show()

    return tree, t

####################################################################################################################
# The event tree gives the event structure for a human in the world. Animals' actions are simplistic in the current
# setting which do not require an event structure.
# Each node of the event tree is an event, where the leaves are simple events. The information for the tree is stores
# in a tree dictionary, which get updated as the world is running. The key for the dictionary is the coordinates of the
# nodes, and the values are the corresponding information of the event on that node.
# For simple events, the values is a trinary value: 0,1,or -1. 0 means the event is completed, 1 means the event has not been
# completed, -1 means just failed the event.
# At each time, the agent knows where he/she is on the event tree.
####################################################################################################################


def main():
    tree, t = initialize_event_tree(config.World.event_tree_file,0)
    print(tree)
    #pos = graphviz_layout(t, prog='dot')
    #nx.draw(t, pos, arrows=False, with_labels=True)
    #plt.show()


#main()