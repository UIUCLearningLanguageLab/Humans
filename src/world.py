import random
from src import config
from src.human import Human
from src.animals import Animal


class World:

    def __init__(self):
        self.human_list = []
        self.animal_list = []
        self.food_list = []
        self.food_threshold = None
        self.hunting_method_list = []
        self.location_list = ['fridge','hut','fire']
        
    def create_humans(self):
        for i in range(config.World.num_humans):
            self.human_list.append(Human(self))

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


