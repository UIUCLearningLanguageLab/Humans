from src import world
from src.display import display

def main():
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    for i in range(10):
        print('running turn {}'.format(i))
        the_world.next_turn()
        print()
    #the_display = display.Display(the_world)
    #the_display.root.mainloop()

main()
