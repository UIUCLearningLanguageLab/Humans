import random
import numpy as np
from Programs.World import config


class Animal:

    ####################################################################################################################
    # define animal class
    # animal object has category, size and an randomized initial location
    # speed is a constant rate that the animal moves
    # vision is a range that within which the animal detect a human's presence
    # For current experiment designs, deer and rabbit are the only two categories of animals may be generated
    ####################################################################################################################

    def __init__(self, world):
        self.world = world
        self.id_number = len(self.world.animal_list)
        self.category = str(np.random.choice(self.world.animal_category, 1, p=[0.5, 0, 0, 0, 0.5, 0, 0, 0, 0,0])[0])

        # animal size refers to the weight of the animal object, which is transfered to many objects in the world, when
        # animals are hunted and butchered, the animal objects are turned into certain amount of food, the amount of
        # food is determined by the size of the animal.
        self.size = random.randint(self.world.animal_size[self.category][0], self.world.animal_size[self.category][1])
        self.remain_size = self.size
        self.x = random.randint(-config.World.world_size/5, config.World.world_size/5)
        self.y = random.randint(-config.World.world_size/5, config.World.world_size/5)
        self.speed = random.uniform(1,5)
        self.vision = random.uniform(30,60)

    ####################################################################################################################
    # define take turn function for an animal.
    # An animal runs if it detect a human, stays at where it is otherwise
    ####################################################################################################################

    def take_turn(self):
        move = 0
        for human in self.world.human_list:
            d = (human.x - self.x)**2 + (human.y - self.y)**2
            if d < self.vision:
                self.run(human,d)
                move = 1
                break
        # if move == 0:
        #   print('Animal{} has nothing to do'.format(self.id_number))

    ####################################################################################################################
    # define run function for an animal.
    # An animal runs away from the human it detect, that is, the opposite direction of which facing the human
    ####################################################################################################################

    def run(self, human,d):
        if d > 0:
            dx = self.x - human.x
            dy = self.y - human.y
            norm = (dx**2+dy**2)**0.5
            dx = dx/norm
            dy = dy/norm
            self.x = self.x + dx*self.speed
            self.y = self.y + dy*self.speed
        else:
            self.x = self.x + self.speed
        # print('Animal{} runs {} from human{}'.format(self.id_number,d,human.id_number))








