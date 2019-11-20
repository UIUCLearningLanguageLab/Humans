import random 
from src import config
import operator
import stats
import csv
import numpy as np
from src import event_tree as et
from src import animals
from src import world
import STN

class Human:

    def __init__(self, world):
        self.world = world
        self.x = random.randint(config.World.tile_size, config.World.world_size - config.World.tile_size)
        self.y = random.randint(config.World.tile_size, config.World.world_size - config.World.tile_size)

        self.id_number = len(self.world.human_list)
        self.name = random.choice(self.world.name_list)

        self.hunger = random.uniform(0.5,0.7)
        self.sleepiness = random.uniform(0,0.5)
        self.thirst = random.uniform(0.7,1)

        self.eat_count = 0
        self.drink_count = 0
        self.sleep_count = 0
        self.idle_count = 0

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
        self.sleep_threshold = 1
        self.hunger_threshold = 1
        self.thirst_threshold = 0.3
        self.focus = None
        self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_file,1)
        self.hunting_method = self.get_hunting_method()
        self.state_change = self.get_state_change()
        self.drive = ['hunger', 'sleepiness', 'thirst']
        self.current_event = ()

        self.dish_list = []
        self.dish_amount = 0

        self.corpus = []
        self.linear_corpus = []

        # action_dict = {'hunt_deer': [A, B, C, D]}

    def get_activated_words(self):
        l = len(self.corpus)
        Steve = STN.Stn(self.corpus)
        #animal = set(the_world.animal_category)
        hunting = {'trapping'}#,'stabbing','shooting','throwing'}
        if len(hunting.intersection(set(Steve.word_list))) == 0:
            words = random.choices(Steve.word_list, k=1)
        else:
            hunting_list = list(set(Steve.word_list).intersection(hunting))
            words = random.choices(hunting_list, k=1)
        return words,Steve

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
        #print('{}{} on status {}'.format(self.name, self.id_number, self.current_event))

        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # specific event functions to write
                event_name = self.event_dict[self.current_event][0]
                if event_name == 'searching':
                    self.searching()
                elif event_name == 'going_to':
                    self.going_to()
                elif event_name in {'gathering', 'butchering', 'cooking', 'eating', 'laying_down', 'sleeping', 'waking_up', 'getting_up',
                                    'getting_drink','drinking','idling', 'washing'}:
                    self.do_it(event_name)
                else:
                    self.hunt(event_name)
                self.generate_language(event_name)
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
                l = len(self.world.human_list)
                if self.world.human_list.index(self) == l-1:
                    self.world.epoch = self.world.epoch + 1
                print(self.hunger, self.sleepiness, self.thirst)
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_file,0)
                self.focus = None
            else:
                self.current_event = self.choose_heir()

        #print(self.eat_count)
        #print(self.sleep_count)
        #print(self.drink_count)
        #print(self.idle_count)

    def generate_language(self,event_name):
        #if self.current_event[0] == 0:
        #    print('{} is hungry.'.format(self.name))
        #elif self.current_event[0] == 1:
        #    print('{} is sleepy.'.format(self.name))
        #elif self.current_event[0] == 2:
        #    print('{} is thirsty.'.format(self.name))

        if self.focus is None or event_name == 'searching':
            #print('{} is {}.'.format(self.name, event_name))
            if event_name is not 'idling':
                self.corpus.append((self.name, event_name))
                self.linear_corpus.append([self.name, event_name])
        else:
            if isinstance(self.focus, animals.Animal):
                focus = self.focus.category
                #print('{} is {} the {}.'.format(self.name, event_name, focus))
            else:
                focus = self.focus
                #print('{} is {} {}.'.format(self.name, event_name, focus))
            self.corpus.append((self.name, (event_name, focus)))
            self.linear_corpus.append([self.name,event_name,focus])

    def choose_heir(self):
        t = self.event_tree
        event_type = self.event_dict[self.current_event][0]
        children = [n for n in t.neighbors(self.current_event)]
        children.sort()
        if event_type == 's':
            index = len(children) - self.event_dict[self.current_event][1]
            self.current_event = children[index]
        elif event_type == 'op':
            score = -float('Inf')
            event = ''
            for child in children:
                new_score = self.compute_scores(child)
                if new_score > score:
                    score = new_score
                    event = child
            self.current_event = event
        else:
            hunting_method_dist = []
            index = self.world.animal_category.index(self.focus.category)
            for child in children:
                hunting_method_dist.append(self.hunting_method[child][index])
            norm = [float(i) / sum(hunting_method_dist) for i in hunting_method_dist]
            self.current_event = children[int(np.random.choice(len(children), 1, p=norm)[0])]
        return self.current_event

    def compute_scores(self, event):
        if self.current_event == ():
            if event == (0, ) and self.hunger >= self.hunger_threshold:
                score = self.hunger
            elif event == (1, ) and self.sleepiness >= self.sleep_threshold:
                score = self.sleepiness
            elif event == (2,) and self.thirst >= self.thirst_threshold:
                score = self.thirst
            elif event == (3,):
                score = 0
            else:
                score = -float('Inf')
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

        elif current_dict[self.current_event][0] in {'op','pp'}:
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

    def searching(self):
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
        self.hunger = self.hunger + movement * self.state_change['searching'][0]

    def going_to(self):
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
        self.hunger = self.hunger + self.speed * self.state_change['going_to'][0]

    def hunt(self,event_name):
        hunting_skill = self.get_hunting_skill()[0]
        print(self.focus)
        index = self.world.animal_category.index(self.focus.category)
        success_rate = hunting_skill[event_name][index]
        num = np.random.choice(2, 1, p=[1-success_rate, success_rate])
        if num == 1:
            self.event_dict[self.current_event][1] = 0
            if event_name != 'chasing' and event_name != 'trapping':
                self.world.animal_list.remove(self.focus)
        else:
            self.event_dict[self.current_event][1] = -1
        self.hunger = self.hunger + self.state_change[event_name][0]

    def do_it(self,event_name):

        if event_name == 'gathering':
            self.hunger = self.hunger + self.focus.size * self.state_change[event_name][0]
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'butchering':
            self.focus.remain_size = self.focus.size/2
            self.world.food_list.append(self.focus)
            self.world.food_stored = self.world.food_stored + self.focus.remain_size
            self.hunger = self.hunger + self.state_change[event_name][0] * self.focus.remain_size
            self.sleepiness = self.sleepiness + self.state_change[event_name][1]
            self.hunger = self.hunger + self.focus.size * self.state_change[event_name][0]
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'cooking':
            amount_need = self.hunger - self.dish_amount
            self.focus = self.world.food_list[0]
            if self.focus.remain_size >= amount_need:
                self.focus.remain_size = self.focus.remain_size - amount_need
                self.dish_amount = self.hunger
                self.world.food_stored = self.world.food_stored - amount_need
                self.event_dict[self.current_event][1] = 0

            else:
                self.dish_amount = self.dish_amount + self.focus.remain_size
                self.world.food_stored = self.world.food_stored - self.focus.remain_size
                self.world.food_list.remove(self.focus)
                self.world.consumption.append((self.focus.category,self.focus.size/2))
                if len(self.world.food_list) == 0:
                    self.event_dict[self.current_event][1] = 0

            self.dish_list.append(self.focus.category)
            if self.event_dict[self.current_event][1] == 0:
                dish_set = set(self.dish_list)
                self.dish_list = list(dish_set)

        elif event_name == 'eating':
            self.focus = self.dish_list.pop()
            self.eat_count = self.eat_count + 1
            if len(self.dish_list) == 0:
                self.hunger = self.hunger - self.dish_amount
                self.sleepiness = self.sleepiness + self.sleepy_rate * self.dish_amount
                self.event_dict[self.current_event][1] = 0

        elif event_name == 'sleeping':
            self.sleep_count = self.sleep_count + 1
            self.sleepiness = 0
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'getting_drink':
            self.focus = random.choice(self.world.drink_category)
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'drinking':
            self.drink_count = self.drink_count + 1
            #print(self.thirst)
            self.thirst = 0
            self.event_dict[self.current_event][1] = 0

        else:
            if event_name == 'idling':
                self.idle_count = self.idle_count + 1
            self.hunger = self.hunger + self.state_change[event_name][0]
            self.event_dict[self.current_event][1] = 0

        self.sleepiness = self.sleepiness + self.state_change[event_name][1]
        self.thirst = self.thirst + self.state_change[event_name][2]


# what is the high level goal (sleep, eat)
# if sleep, do the sleep steps
# if eat, is there gathered food available


# within those, what are specific goal (eat

