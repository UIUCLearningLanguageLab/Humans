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

        self.speed = None
        self.vision = None
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
        self.state_change = self.get_state_change()


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

    def get_state_change(self):
        state_change = {}
        with open('state_change.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line = 0
            for row in csv_reader:
                if line > 0:
                    copy = []
                    for item in row[1:]:
                        copy.append(eval(item))
                    state_change[row[0]] = np.asarray(copy)
                    line = line + 1
        return state_change


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
                elif event_name in {'gather', 'butcher', 'cook', 'eat', 'lay_down', 'asleep', 'wake_up', 'get_up'}:
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
                score = self.hunger
            else:
                score = self.world.food_stored
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
        region = random.choice(['east', 'west', 'north', 'south'])
        game_list = []
        original_position = (self.x, self.y)
        if region == 'east':
            for animal in self.world.animal_list:
                if animal.x > abs(animal.y):
                    game_list.append(animal)
            self.x = config.World.world_size/4
            self.y = 0
        elif region == 'west':
            for animal in self.world.animal_list:
                if -animal.x > abs(animal.y):
                    game_list.append(animal)
            self.x = -config.World.world_size / 4
            self.y = 0
        elif region == 'north':
            for animal in self.world.animal_list:
                if animal.y > abs(animal.x):
                    game_list.append(animal)
            self.y = config.World.world_size / 4
            self.x = 0
        else:
            for animal in self.world.animal_list:
                if -animal.y > abs(animal.x):
                    game_list.append(animal)
            self.y = -config.World.world_size / 4
            self.x = 0
        movement = ((self.x - original_position[0])**2 + (self.y - original_position[1])**2)**0.5

        if len(game_list) > 0:
            choice = game_list[0]
            d_min = (choice.x - self.x)**2 + (choice.y - self.y)**2
            for game in game_list[1:]:
                d = (game.x - self.x)**2 + (game.y - self.y)**2
                if d < d_min:
                    choice = game
                    d_min = d
            self.focus = choice
            self.event_dict[self.current_event][1] = 0
        self.hunger = self.hunger + movement * self.state_change['search'][0]

    def go_to(self):
        dx = self.x - self.focus.x
        dy = self.y - self.focus.y
        norm = (dx ** 2 + dy ** 2) ** 0.5
        dx = dx / norm
        dy = dy / norm
        self.x = self.x + dx * self.speed
        self.y = self.y + dy * self.speed
        d = ((self.x - self.focus.x)**2 + (self.y - self.focus.y)**2)**0.5
        if d < self.vision:
            self.event_dict[self.current_event][1] = 0
        self.hunger = self.hunger + self.speed * self.state_change['go_to'][0]

    def hunt(self):
        event_name = self.event_dict[self.current_event][0]
        hunting_skill = self.get_hunting_skill()[0]
        index = self.world.animal_category.index(self.focus.category)
        success_rate = hunting_skill[event_name][index]
        num = np.random.choice(2, 1, p=[1-success_rate, success_rate])
        if num == 1:
            self.event_dict[self.current_event][1] = 0
        self.hunger = self.hunger + self.state_change[event_name][0]

    def do_it(self):
        event_name = self.current_event[self.current_event][0]

        if event_name == 'butcher':
            if self.focus:
                self.world.food_stored = self.world.food_stored + self.focus.size
                self.hunger = self.hunger + self.state_change[event_name][0]* self.focus.size
                self.sleepiness = self.sleepiness + self.state_change[event_name][1]

            self.focus = None


        if event_name == 'eat':
            self.food_stored = self.food_stored
            self.sleepiness = self.sleepiness + self.hunger
            self.hunger = 0


        self.hunger = self.hunger + self.state_change[event_name][0]
        self.sleepiness = self.sleepiness + self.state_change[event_name][1]

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
