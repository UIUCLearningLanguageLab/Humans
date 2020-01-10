import random
from src import config
from src.human import Human
from src.animals import Animal


class World:
    ################################################################################################################
    # define the world object, detailed setting of the world
    # the world is first generated, and then humans and animals
    # the world generated serves as a required attribute for the humans and animals, it indicate which world is the
    # human and the animal living in

    # all agents in the world conform to the world principals listed below
    ################################################################################################################

    def __init__(self):

        #  type of the predicates(simple event), the number refers to the argument it takes,
        self.simple_event_dict = {'search': 1, 'go_to': 1, 'trap': 2, 'catch': 2, 'chase': 2, 'stab': 2, 'shoot': 2,
                                  'throw_at': 2, 'gather': 2, 'butcher': 2, 'cook': 2, 'eat': 2, 'lay_down': 1,
                                  'asleep': 1, 'wake_up': 1, 'get_up': 1}

        # possible names of the humans in the world, which are the names of lab members in Learning & Language Lab
        self.name_list = ['Jessica','Jon','Anastasia','Phil','Andrew','Lin Khern','Emily','Shufan','Jacki','Katherine']
        self.human_list = []

        # possible animal categories
        self.animal_category = ['rabbit', 'squirrel', 'fox', 'mouflon', 'deer', 'boar', 'ibex', 'bison', 'mammoth',
                                'auroch']

        # the range of size of an animal object in each category
        self.animal_size = {'rabbit':(3,5), 'squirrel':(1,2), 'fox':(9,19), 'mouflon':(50,100), 'deer':(180,260),
                            'boar':(88,110), 'ibex':(80,150), 'bison':(1800,2200), 'mammoth':(8000,10000),
                                'auroch':(1000,2000)}
        self.animal_list = []

        # possible fruit categories
        self.fruit_category = ['apple', 'peach', 'pear']

        # possible drink categories
        self.drink_category = ['water', 'juice', 'beer']

        self.food_list = []
        self.food_stored = 0
        self.consumption = []
        self.hunting_method_list = []

        # possible locations in the world
        self.location_list = ['fridge', 'hut', 'fire','orchard']

        # an epoch is a period staring from the initialization of the event tree of a human, until the event tree get
        # completed, this is well defined since currently only one human is generated in the world
        self.epoch = 0

    def create_humans(self):  # humans are generated once the world is there, the current setting is a hunter-gatherer's
        # world
        for i in range(config.World.num_humans):
            self.human_list.append(Human(self))
            if config.World.event_tree_file == 'src/event_tree.txt':
                self.human_list[i].get_hunting_method()

    def create_animals(self):
        for i in range(config.World.num_animals):
            self.animal_list.append(Animal(self))

    def next_turn(self): # human and animals take turns in order
        for human in self.human_list:
            human.take_turn()

        for animal in self.animal_list:
            animal.take_turn()

        # hunt
        #   location: where animal is
        #   instruments: bows, arrows, spears, hand axe, trap

        # cooking
        #   # location: hearth
        #   instruments: fire, spear

        # sleep
        #   location: hut
        #   instruments: blanket


