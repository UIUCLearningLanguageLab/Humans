from src import world
from src import config
import STN_analysis
import STN
import random
from src.display import display


def running_world():
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    for i in range(config.World.num_turn):
        the_world.next_turn()
    #the_display = display.Display(the_world)
    #the_display.root.mainloop()
    num_consumed_animal = config.World.num_animals - len(the_world.animal_list)
    print('{} animals consumed.'.format(num_consumed_animal))
    print(the_world.consumption)
    print('{} epochs passed'.format(the_world.epoch))
    return the_world


def activation_dispersion_measure():
    the_world = running_world()
    for human in the_world.human_list:
        words,Steve = human.get_activated_words()
        linear_Doug = STN.Dg(human.linear_corpus)
        Steve.plot_network()
        STN_analysis.activation_spreading_analysis(Steve, words)
        STN_analysis.activation_spreading_analysis(linear_Doug, words)


activation_dispersion_measure()