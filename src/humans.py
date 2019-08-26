import random
from src import config


class Human:

    def __init__(self, world):

        self.x = random.randint(config.World.tile_size, config.World.world_size - config.World.tile_size)
        self.y = random.randint(config.World.tile_size, config.World.world_size - config.World.tile_size)

        self.id_number = None
        self.name = None
        self.hunger = None
        self.sleepiness = None
        self.fatigue = None
        self.energy = None
        self.health = None
        self.max_speed = None
        self.awake = True
        self.age = None
        self.size = None
        self.strength = None
        self.intelligence = None
        self.food_threshold = None
        self.carry_threshold = None
        self.sleep_threshold = None
        self.hunger_threshold = None

        self.world = world


    def take_turn(self):
        x_delta =  random.randint(-10, 10)
        y_delta = random.randint(-10, 10)
        if config.World.tile_size < self.x + x_delta < config.World.world_size-config.World.tile_size:
            self.x += x_delta
        if config.World.tile_size < self.y + y_delta < config.World.world_size-config.World.tile_size:
            self.y += y_delta
            

    def get_hunt_success_probs(self, animal):
        best_action = None
        prob = 0
        stab_prob = 0
        shoot_prob = 0
        throw_prob = 0
        trap_prob = 0
        return best_action, prob

    def stay(self):
        raise NotImplementedError

    def go_to(self,location):
        raise NotImplementedError

    def check_food(self):
        food_amount = 0

        food_list = self.world.food_list


        return food_amount

    def butcher(self,food):
        raise NotImplementedError

    def cook(self,food):
        raise NotImplementedError

    def eat(self,food):
        raise NotImplementedError

    def sleep(self):
        raise NotImplementedError


    # decide what to eat based on the extent of hunger and what is there in the fridge,
    # choose food so that the energy contained in food should be no less than the sum of hunger and the energy consumed
    # to conduct the cooking-eating sequence, if food is enough, return the food and amount, otherwise: if the energy
    # is lower than threshold, return all food restored and amount, if energetic, return 0

    def decide_food_to_eat(self):

        food_decided = []
        return food_decided

    # go and look for animal to hunt, if nothing is found, return 0, else return one animal represented by
    # (id, category, size, location)

    def look_for_animal(self):
        game = 0

        return game


    # calculate the expected energy gain from hunting the animal, where E(gain) = E(game)-E(killing)-E(carrying),
    # E(game)= sum(energy of all animal found); E(carrying)= energy consumed to carry the animals;


    def calculate_gain(self, game):
        gain = 0
        return gain



    # decide whether to kill the found animal, based on the expected energy consumed to conduct the killing-carrying sequence
    # against the gain, decide what animals to hunt so that satisfying the hunger return a list of Animal , otherwise, return 0.

    def search(self):
        game = self.look_for_animal()
        if game:
            game_value = self.calculate_gain(game)
            if game_value > self.hunger:
                animal_found = game
            else:
                animal_found = 0
        else:
            animal_found = game

        return animal_found


    def choose_killing_method(self, animal_found):
        raise NotImplementedError


    def kill(self, animal, method):
        raise NotImplementedError


    def carry_to_fridge(self, animal_found):
        raise NotImplementedError

    def get_up(self):
        raise NotImplementedError




    def take_action(self):


        # given calculations of expectation about actions satisfying drives, choose an action
        if self.awake:
            # when one is awake, he/she does three things: go sleeping, go get food, or just stay
            # if sleepiness pass the threshold, then go sleeping, otherwise, if hunger pass some
            # threshold, go get food, finally, if the neither sleepy nor hungry, then just stay.

            if self.sleepiness >= self.sleep_threshold:
                self.sleep()

            else:
                # fisrt go check how much food left, no matter hungry or not
                self.go_to('fridge')
                food_amount = self.check_food()
                if food_amount >= self.food_threshold:

                    # in the case that food is enough(below the threshold), do cooking-eating sequence if hungry
                    # do nothing if not

                    if self.hunger < self.hunger_threshold:
                        self.stay()
                    else:
                        to_eat_list = self.decide_food_to_eat()
                        for animal in to_eat_list:
                            if not animal.buthcered:
                                self.butcher(animal)
                            self.cook(animal)
                            self.eat(animal)

                else:
                    # in the case that food is not enough, should go get food.

                    # Yet, if hungry, should first get something to eat
                    # get a full meal if enough food, otherwise, eat all stored food

                    to_eat_list = self.decide_food_to_eat()
                    if to_eat_list:
                        for animal in to_eat_list:
                            if not animal.buthcered:
                                self.butcher(animal)
                            self.cook(animal)
                            self.eat(animal)

                    # If not hungry, then go out and get food (hunt)
                    else:
                        energy_gain = 0
                        hunt_goal = self.food_threshold - food_amount
                        animal_killed = []
                        animal_processed = []
                        # do the search-go-kill-(butcher)-carry sequence until goal met
                        while energy_gain < hunt_goal:
                            animal_found = []
                            while len(animal_found) == 0:
                                animal_found = self.search()
                            method = self.choose_killing_method(animal_found)
                            animal = animal_found[1]
                            animal_location = animal_found[3]
                            self.go_to(animal_location)
                            result = self.kill(animal, method)
                            if result:
                                animal_killed.append(animal)
                                energy_gain = + animal[2]
                                if animal[2] > self.carry_threshold:
                                    butchered_animal = self.butcher(animal)
                                    animal_processed.append(butchered_animal)
                                else:
                                    animal_processed.append(animal)
                                self.carry_to_fridge(animal)

                        # update the food storage
                        for animal in animal_processed:
                            self.world.food_list.append(animal)



        else:
            if self.sleepiness <= self.sleep_threshold:
                self.get_up()


            
# what is the high level goal (sleep, eat)
# if sleep, do the sleep steps
# if eat, is there gathered food available



# within those, what are specific goal (eat
