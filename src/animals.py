import random


class Animal:

    def __init__(self, world):
        self.world = world
        self.id_number = len(self.world.animal_list)
        self.category = random.choice(self.world.animal_category)
        self.size = random.randint(self.world.animal_size[self.category][0], self.world.animal_size[self.category][1])
        self.location = None





