from src import world
from src import config
import STN_analysis
import STN
import random
from src.display import display


def main():
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    for i in range(config.World.num_turn):
        the_world.next_turn()
    #the_display = display.Display(the_world)
    #the_display.root.mainloop()
    for human in the_world.human_list:
        l = len(human.corpus)
        Steve = STN.Stn(human.corpus[:round(l/50)])
        words = random.choices(Steve.word_list, k=1)
        STN_analysis.activation_spreading_analysis(Steve, words)
        moments = [round(l/20), round(l/10), round(l/5), round(l/2), l]
        for moment in moments:
            Steven = STN.Stn(human.corpus[:moment])
            STN_analysis.activation_spreading_analysis(Steven,words)
main()
