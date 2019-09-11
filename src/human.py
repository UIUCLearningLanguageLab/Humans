import random 
from src import config
import operator
import stats
import csv
import numpy as np
from src import event_tree as et


class Human:

    def __init__(self, world):
        self.world = world
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
        self.focus = None
        self.event_dict, self.event_tree = et.initialize_event_tree('event_tree.txt')
        self.hunting_method = self.get_hunting_method()

        self.drive_dict = {'hunger': 0, 'sleepiness': 0}
        self.drive_values = [0, 0]

        self.current_drive = ['hunger', 'hunt_deer', 'shoot']
        self.current_event = None

        # action_dict = {'hunt_deer': [A, B, C, D]}

    def get_hunting_method(self):
        t = self.event_tree
        hunting_skill, length = self.get_hunting_skill()
        hunting_method = {}
        method_list = [(0,0,0,1,0), (0,0,0,1,1), (0,0,0,1,2), (0,0,0,1,3)]
        for method in method_list:
            skills = [n for n in t.neighbors(method)]
            success_rate = np.ones(length)
            for skill in skills:
                skill_name = self.event_dict[skill][0]
                skill_rate = hunting_skill[skill_name]
                success_rate = np.multiply(success_rate, skill_rate)
            hunting_method[method] = success_rate
        return hunting_method

    def get_hunting_skill(self):
        hunting_skill = {}
        with open('random_sampling.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line = 0
            for row in csv_reader:
                if line == 0:
                    length = len(row)-2
                else:
                    copy = []
                    for item in row[1:-1]:
                        copy.append(eval(item))
                    hunting_skill[row[0]] = np.asarray(copy)
                    line = line + 1
        return hunting_skill, length

    def take_turn(self):
        t = self.event_tree
        self.compute_status()
        status = self.event_dict[self.current_event][1]

        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # specific event functions to write
                event_name = self.event_dict[self.current_event][0]
                if event_name == 'search':
                    self.search()
                elif event_name == 'go_to':
                    self.go_to()
                elif event_name  in {'gather', 'butcher', 'cook', 'eat', 'lay_down', 'asleep', 'wake_up', 'get_up'}:
                    self.do_it()
                else:
                    self.hunt()
            else:
                self.current_event = self.current_event[:len(self.current_event)-1]

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
        sorted_children = children.sort()
        if event_type == 's':
            index = len(children) - self.event_dict[self.current_event][1]
            self.current_event = sorted_children[index]
        else:
            score = -float('Inf')
            event = ''
            for child in children:
                new_score = self.compute_scores(child)
                if new_score > score:
                    score = new_score
                    event = child
            self.current_event = event
        return self.current_event

    def compute_scores(self, event):
        if self.current_event == ():
            if event == (0, ):
                score = self.hunger
            else:
                score = self.sleepiness
        elif self.current_event == (0, 0):
            if event == (0, 0, 0):
                score = -self.world.food_stored
            else:
                score = 0
        else:
            index = self.world.animal_category.index(self.focus)
            score = self.hunting_method[event][index]

        return score

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

    def hunt(self):
        print('hunt')

    def do_it(self):
        print('do_it')

    def choose_killing_method(self, animal_found):
        raise NotImplementedError

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
