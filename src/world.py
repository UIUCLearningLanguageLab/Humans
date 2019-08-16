

class World:

    def __init__(self):
        running = False
        self.human_list = []
        self.animal_list = []

        while running:
            for human in self.human_list:
                human.take_action()
            for animal in self.animal_list:
                animal.take_action()

        # some humans


        # some animals


        # hunt
        #   location: where animal is
        #   instruments: bows, arrows, spears, handaxe, trap

        # cooking
        #   # location: hearth
        #   instruments: fire, spear

        # sleep
        #   location: hut
        #   instruments: blanket

