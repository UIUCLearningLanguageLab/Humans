import random
from src import config
import operator
import stats
from src import event_tree as et


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
        self.event_dict, self.event_tree = et.initialize_event_tree('event_tree.txt')

        self.drive_dict = {'hunger': 0, 'sleepiness': 0}
        self.drive_values = [0, 0]

        self.current_drive = ['hunger', 'hunt_deer', 'shoot']
        self.current_event = None

        # action_dict = {'hunt_deer': [A, B, C, D]}

    def take_turn(self):
        self.compute_status()

    def compute_status(self):
        T = self.event_tree
        current_dict = self.event_dict

        if current_dict[self.current_event][0] == "s":
            score = 0
            for event in T.neighbors(self.current_event):
                if dict[event][1] > 0:
                    score = score + 1
            self.event_dict[self.current_event][1] = score

        elif current_dict[self.current_event][0] == 'p':
            for event in T.neighbors(self.current_event):
                if dict[event][1] == 0:
                    self.event_dict[self.current_event][1] = 0
                    break

    def move(self):
        x_delta = random.randint(-10, 10)
        y_delta = random.randint(-10, 10)
        if config.World.tile_size < self.x + x_delta < config.World.world_size - config.World.tile_size:
            self.x += x_delta
        if config.World.tile_size < self.y + y_delta < config.World.world_size - config.World.tile_size:
            self.y += y_delta

    def search(self):
        raise NotImplementedError

    def go_to(self, location):
        raise NotImplementedError

    def choose_killing_method(self, animal_found):
        raise NotImplementedError

    def get_hunt_success_probs(self, animal):
        best_action = None
        prob = 0
        stab_prob = 0
        shoot_prob = 0
        throw_prob = 0
        trap_prob = 0
        return best_action, prob

    def trap(self, animal):
        raise NotImplementedError

    def catch(self, animal):
        raise NotImplementedError

    def chase(self, animal):
        raise NotImplementedError

    def stab(self, animal):
        raise NotImplementedError

    def shoot(self, animal):
        raise NotImplementedError

    def throw_at(self, animal):
        raise NotImplementedError

    def gather(self, animal):
        raise NotImplementedError

    def butcher(self, food):
        raise NotImplementedError

    def cook(self, food):
        raise NotImplementedError

    def eat(self, food):
        raise NotImplementedError

    def lay_down(self):
        raise NotImplementedError

    def fall_asleep(self):
        raise NotImplementedError

    def wake_up(self):
        raise NotImplementedError

    def get_up(self):
        raise NotImplementedError

# what is the high level goal (sleep, eat)
# if sleep, do the sleep steps
# if eat, is there gathered food available


# within those, what are specific goal (eat
