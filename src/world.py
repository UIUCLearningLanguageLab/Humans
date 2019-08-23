import random
from src import config


class World:

    def __init__(self):
        self.human_list = []
        self.create_humans()

    def create_humans(self):
        for i in range(config.World.num_humans):
            self.human_list.append(Human())

    def next_turn(self):
        for human in self.human_list:
            human.take_turn()

        # hunt
        #   location: where animal is
        #   instruments: bows, arrows, spears, handaxe, trap

        # cooking
        #   # location: hearth
        #   instruments: fire, spear

        # sleep
        #   location: hut
        #   instruments: blanket

