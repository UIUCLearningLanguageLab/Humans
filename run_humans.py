from src import world
from src import config
import STN_analysis
import STN
import HAL_analysis
import numpy as np
import random
from src.display import display
VERBOSE = False
window_types = ['forward','backward','summed']
window_sizes = [2,3,4,5,6,7,8,9]
window_weights = ['linear','flat']



def running_world():
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    for i in range(config.World.num_turn):
        the_world.next_turn()
    #the_display = display.Display(the_world)
    #the_display.root.mainloop()
    num_consumed_animal = config.World.num_animals - len(the_world.animal_list)
    if VERBOSE:
        print('{} animals consumed.'.format(num_consumed_animal))
        print(the_world.consumption)
        print('{} epochs passed'.format(the_world.epoch))
    return the_world


def activation_dispersion_measure():
    the_world = running_world()
    for human in the_world.human_list:
        Steve = human.get_activated_words()[1]
        linear_Doug = STN.Dg(human.linear_corpus)
        linear_corpus = linear_Doug.linear_corpus
        if VERBOSE:
            Steve.plot_network()
        source = 'waiting'
        target = ['rabbit','deer','water']
        sl_steve = STN_analysis.activation_spreading_analysis(Steve, source, target)
        sl_doug = STN_analysis.activation_spreading_analysis(linear_Doug, source, target)
        if VERBOSE:
            print('semantic relatedness by STN:')
            print(sl_steve)
            print('semantic relatedness by Distributional Graph')
            print(sl_doug)
        recording_matrix = np.zeros((len(window_sizes) + 1, len(window_weights) * len(window_types)))
        for i in range(len(window_sizes)):
            for j in range(len(window_weights)):
                for k in range(len(window_types)):
                    encoding = {'window_size':window_sizes[i], 'window_weight':window_weights[j],
                                'window_type':window_types[k]}
                    sl_hal = HAL_analysis.get_cos_sim(linear_corpus,source,target,encoding)
                    if sl_hal[target[0]] > sl_hal[target[1]] > sl_hal[target[2]]:
                        recording_matrix[i][j*len(window_types)+k] = 1
                    if VERBOSE:
                        print(encoding)
                        print(sl_hal)
        if sl_steve[target[0]] > sl_steve[target[1]] > sl_steve[target[2]]:
            recording_matrix[len(window_sizes)][0] = 1
        if sl_doug[target[0]] > sl_doug[target[1]] > sl_doug[target[2]]:
            recording_matrix[len(window_sizes)][1] = 1
        return recording_matrix


def run_experiments(run_times):
    performance_matrix = np.zeros((len(window_sizes)+1,len(window_weights)*len(window_types)))
    for i in range(run_times):
        performance_matrix += activation_dispersion_measure()
        if i % 5 == 0:
            print('{} turns run'.format(i))
    performance_matrix = performance_matrix/run_times
    print(performance_matrix)

run_experiments(100)