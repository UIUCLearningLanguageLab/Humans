import random
from src import config
from src.human import Human
from src.animals import Animal


class World:

    def __init__(self):
        self.simple_event_dict = {'search': 1, 'go_to': 1, 'trap': 2, 'catch': 2, 'chase': 2, 'stab': 2, 'shoot': 2,
                                  'throw_at': 2, 'gather': 2, 'butcher': 2, 'cook': 2, 'eat': 2, 'lay_down': 1,
                                  'asleep': 1, 'wake_up': 1, 'get_up': 1}
        self.human_list = []
        self.animal_category = ['rabbit', 'squirrel', 'fox', 'mouflon', 'deer', 'boar', 'ibex', 'bison', 'mammoth',
                                'auroch']
        self.drink_category = ['water']
        self.animal_size = {'rabbit':(3,5), 'squirrel':(1,2), 'fox':(9,19), 'mouflon':(50,100), 'deer':(180,260),
                            'boar':(88,110), 'ibex':(80,150), 'bison':(1800,2200), 'mammoth':(8000,10000),
                                'auroch':(1000,2000)}
        self.animal_list = []
        self.food_list = []
        self.food_stored = 0
        self.hunting_method_list = []
        self.location_list = ['fridge', 'hut', 'fire']
        
    def create_humans(self):
        for i in range(config.World.num_humans):
            self.human_list.append(Human(self))
            if config.World.event_tree_file == 'src/event_tree.txt':
                self.human_list[i].get_hunting_method()

    def create_animals(self):
        for i in range(config.World.num_animals):
            self.animal_list.append(Animal(self))

    def next_turn(self):
        for human in self.human_list:
            human.take_turn()

        for animal in self.animal_list:
            animal.take_turn()

        # hunt
        #   location: where animal is
        #   instruments: bows, arrows, spears, handaxe, trap

        # cooking
        #   # location: hearth
        #   instruments: fire, spear

        # sleep
        #   location: hut
        #   instruments: blanket


