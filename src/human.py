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
        t = self.event_tree
        self.compute_status()
        status = self.event_dict[self.current_event][1]

        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # specific event functions to write
                print(self.event_dict[self.current_event][0])
            else:
                self.current_event = self.current_event[len(self.current_event)-1]

        elif t.in_degree(self.current_event) != 0:  # currently on branch
            if status == 0:
                self.current_event = self.current_event[len(self.current_event)-1]
            else:
                self.current_event = self.choose_heir()

        else:  # currently on the root
            if status == 0:
                self.event_dict, self.event_tree = et.initialize_event_tree('event_tree,txt')
            else:
                self.current_event = self.choose_heir()

    def choose_heir(self):
        t = self.event_tree
        event_type = self.event_dict[self.current_event][0]
        children = [n for n in t.neighbors(self.current_event)]
        num = len(children)
        if event_type == 's':
            index = num - self.event_dict[self.current_event][1]
            self.current_event = self.current_event + (index,)
        else:  # choice function to write
            scores = []
            for child in children:
                score = self.compute_scores(child)
                scores.append(score)
            self.current_event = self.make_decision(scores)
        return self.current_event

    def compute_scores(self,event):
        if self.current_event == ():
            score = 1
        elif self.current_event == (0,0):
            score = 2
        elif self.current_event == (0,0,0,1):
            score = 3
        else:
            score = 4
        return score

    def make_decision(self,event,scores):
        return self.current_event


    def compute_status(self):
        t = self.event_tree
        current_dict = self.event_dict

        if current_dict[self.current_event][0] == "s":
            score = 0
            for event in t.neighbors(self.current_event):
                if current_dict[event][1] > 0:
                    score = score + 1
            self.event_dict[self.current_event][1] = score

        elif current_dict[self.current_event][0] == 'p':
            for event in t.neighbors(self.current_event):
                if current_dict[event][1] == 0:
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
