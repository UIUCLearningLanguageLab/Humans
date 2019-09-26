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

        self.id_number = len(self.world.human_list)
        self.name = None

        self.hunger = random.uniform(0.5,0.7)
        self.sleepiness = random.uniform(0,0.5)
        self.thirst = random.uniform(0.7,1)

        self.sleepy_rate = 0.05
        self.speed = random.uniform(10,20)
        self.vision = random.uniform(50,100)
        self.fatigue = None
        self.energy = None
        self.health = None
        self.max_speed = None
        self.awake = True
        self.age = random.randint(0,100)
        self.size = None
        self.strength = None
        self.intelligence = None
        self.food_threshold = None
        self.carry_threshold = None
        self.sleep_threshold = None
        self.hunger_threshold = None
        self.focus = None
        self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_file)
        self.hunting_method = None
        self.state_change = self.get_state_change()
        self.current_drive = ['hunger', 'hunt_deer', 'shoot']
        self.current_event = ()

        self.dish_list = []
        self.dish_amount = 0

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
        self.hunting_method = hunting_method

    @staticmethod
    def get_hunting_skill():
        hunting_skill = {}
        with open('src/random_sampling.csv') as csv_file:
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

    @staticmethod
    def get_state_change():
        state_change = {}
        with open('src/state_change.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line = 0
            for row in csv_reader:
                if line > 0:
                    copy = []
                    for item in row[1:]:
                        copy.append(eval(item))
                    state_change[row[0]] = np.asarray(copy)
                line = line + 1
        #print(state_change)
        return state_change

    def take_turn(self):
        t = self.event_tree
        self.compute_status()
        status = self.event_dict[self.current_event][1]
        print('Human{} on status {}'.format(self.id_number,self.current_event))

        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # specific event functions to write
                event_name = self.event_dict[self.current_event][0]
                if event_name == 'search':
                    self.search()
                elif event_name == 'go_to':
                    self.go_to()
                elif event_name in {'gather', 'butcher', 'cook', 'eat', 'lay_down', 'asleep', 'wake_up', 'get_up',
                                    'get_water','drink','null'}:
                    self.do_it()
                else:
                    self.hunt()
            elif status == 0:
                self.current_event = self.current_event[:len(self.current_event)-1]
            else:
                self.current_event = ()

        elif t.in_degree(self.current_event) != 0:  # currently on branch
            if status == 0:
                self.current_event = self.current_event[:len(self.current_event)-1]
            else:
                self.current_event = self.choose_heir()

        else:  # currently on the root
            if status == 0:
                print('epoch finished')
                print(self.hunger, self.sleepiness, self.thirst)
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_file)
            else:
                self.current_event = self.choose_heir()


    def choose_heir(self):
        t = self.event_tree
        event_type = self.event_dict[self.current_event][0]
        children = [n for n in t.neighbors(self.current_event)]
        children.sort()
        if event_type == 's':
            index = len(children) - self.event_dict[self.current_event][1]
            self.current_event = children[index]
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
            elif event == (1, ):
                score = self.sleepiness
            else:
                score = self.thirst
        elif self.current_event == (0, 0):
            if event == (0, 0, 0):
                score = self.hunger
            else:
                score = self.world.food_stored
        else:
            index = self.world.animal_category.index(self.focus.category)
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
        self.x = self.x - dx * self.speed
        self.y = self.y - dy * self.speed
        d = ((self.x - self.focus.x)**2 + (self.y - self.focus.y)**2)**0.5
        if d < self.vision:
            self.event_dict[self.current_event][1] = 0
        self.hunger = self.hunger + self.speed * self.state_change['go_to'][0]

    def hunt(self):
        event_name = self.event_dict[self.current_event][0]
        hunting_skill = self.get_hunting_skill()[0]
        print(self.focus)
        index = self.world.animal_category.index(self.focus.category)
        success_rate = hunting_skill[event_name][index]
        num = np.random.choice(2, 1, p=[1-success_rate, success_rate])
        if num == 1:
            self.event_dict[self.current_event][1] = 0
            if event_name != 'chase' and event_name != 'trap':
                self.world.animal_list.remove(self.focus)
        else:
            self.event_dict[self.current_event][1] = -1
        self.hunger = self.hunger + self.state_change[event_name][0]

    def do_it(self):
        event_name = self.event_dict[self.current_event][0]
        print(event_name)

        if event_name == 'gather':
            self.hunger = self.hunger + self.focus.size * self.state_change[event_name][0]
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'butcher':
            self.world.food_list.append(self.focus)
            self.world.food_stored = self.world.food_stored + self.focus.size
            self.hunger = self.hunger + self.state_change[event_name][0] * self.focus.size
            self.sleepiness = self.sleepiness + self.state_change[event_name][1]
            self.hunger = self.hunger + self.focus.size * self.state_change[event_name][0]
            self.focus = None
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'cook':
            amount_need = self.hunger - self.dish_amount
            self.focus = self.world.food_list[0]
            max_size = self.focus.size
            for food in self.world.food_list:
                if food.size > max_size:
                    self.focus = food
                    if food.size > amount_need:
                        break
            if self.focus.size >= amount_need:
                self.focus.size = self.focus.size - amount_need
                self.dish_amount = self.hunger
                self.world.food_stored = self.world.food_stored - amount_need
                self.event_dict[self.current_event][1] = 0

            else:
                self.dish_amount = self.dish_amount + self.focus.size
                self.world.food_stored = self.world.food_stored - self.focus.size
                self.world.food_list.remove(self.focus)
                if len(self.world.food_list) == 0:
                    self.event_dict[self.current_event][1] = 0

            self.dish_list.append(self.focus.category)
            if self.event_dict[self.current_event][1] == 0:
                dish_set = set(self.dish_list)
                self.dish_list = list(dish_set)
            self.focus = None

        elif event_name == 'eat':
            self.focus = self.dish_list.pop()
            if len(self.dish_list) == 0:
                self.hunger = self.hunger - self.dish_amount
                self.sleepiness = self.sleepy_rate * self.dish_amount
                self.event_dict[self.current_event][1] = 0
            self.focus = None

        elif event_name == 'asleep':
            if self.sleepiness < self.hunger or self.sleepiness < self.thirst:
                self.event_dict[self.current_event][1] = 0

        elif event_name == 'drink':
            self.focus = random.choice(self.world.drink_category)
            print(self.thirst)
            self.event_dict[self.current_event][1] = 0
            self.focus = None

        else:
            self.hunger = self.hunger + self.state_change[event_name][0]
            self.event_dict[self.current_event][1] = 0

        self.sleepiness = self.sleepiness + self.state_change[event_name][1]
        self.thirst = self.thirst + self.state_change[event_name][2]


# what is the high level goal (sleep, eat)
# if sleep, do the sleep steps
# if eat, is there gathered food available


# within those, what are specific goal (eat
