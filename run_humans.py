from src import world
from src.display import display

def main():
    the_world = world.World()
    the_world.create_humans()
    the_world.create_animals()
    the_display = display.Display(the_world)
    the_display.root.mainloop()

main()
