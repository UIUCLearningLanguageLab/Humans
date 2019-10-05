from src import world
from src import config
import STN_analysis
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
        print(human.corpus)
        STN_analysis.analysis(human.corpus)
main()
