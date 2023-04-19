import csv
import random

import numpy as np

from Programs.Graphical_Models import STN
from Programs.World import animals
from Programs.World import config
from Programs.World import event_tree as et

VERBOSE = False


class Human:
    ####################################################################################################################
    # define human class
    ####################################################################################################################

    def __init__(self, world, name):
        self.world = world  # a human object lives in a certain world

        # a human has basic properties including position, name
        self.x = random.randint(config.World.tile_size, config.World.world_size - config.World.tile_size)
        self.y = random.randint(config.World.tile_size, config.World.world_size - config.World.tile_size)
        self.id_number = len(self.world.human_list)
        self.name = name
        self.type = 'human'
        self.animal_type = 'human'
        self.category = 'human'

        # a human has basic drives which motivate their actions, currently including hunger, sleepiness and thirst
        self.hunger = random.uniform(0.5,0.7)
        self.sleepiness = random.uniform(0,0.5)
        self.thirst = random.uniform(0.7,1)

        # threshold for drives any abilities
        self.food_threshold = None
        self.carry_threshold = None
        self.sleep_threshold = 1
        self.hunger_threshold = 2
        self.thirst_threshold = 0.3

        # count how many times a human has acted fullfilling each drive
        self.eat_count_meal = 0
        self.eat_count_fruit = 0
        self.eat_count_nut = 0
        self.eat_count_plant = 0
        self.drink_count = 0
        self.sleep_count = 0
        self.idle_count = 0

        # the rate for getting sleepy while idling
        self.sleepy_rate = 0.05

        # basic physical attributes, vision is the range for detecting an animal
        self.speed = random.uniform(50,100)
        self.vision = random.uniform(50,100)

        # other attributes which may be included in more complex world designs
        self.fatigue = None
        self.energy = None
        self.health = None
        self.max_speed = None
        self.awake = True
        self.alive = True
        self.age = random.randint(0,100)
        self.size = None
        self.strength = None
        self.intelligence = None

        ################################################################################################################
        # event related attributes
        ################################################################################################################

        self.focus = None  # when human are involved in transitive(ditransitive) events, focus is the patient, otherwise
        # it is none
        self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_human, 0)  # generate the
        # event tree structure which human obeys.
        self.destination = None # where to go
        self.food_target = self.get_target() # get the food target
        self.hunting_method = self.get_hunting_method()  # get the hunting method, which is a distribution over the
        #print(self.hunting_method)
        # methods can be used for hunting
        self.state_change = self.get_state_change()  # get the change of drive levels for all actions.
        self.drive = ['hunger', 'sleepiness', 'thirst']
        self.current_event = ()  # where the human is on the event tree at hte moment

        # when cooking, foods are turned into some number of dishes, dish_list record the dishes remained, dish amount
        # record the total amount (of energy) of the dishes.
        self.food_list = []
        self.food_stored = 0
        self.dish_list = []
        self.dish_amount = 0

        ################################################################################################################
        # corpus related attributes
        ################################################################################################################
        self.corpus = []  # corpus of parsed structures, prepared for models requiring syntax
        self.linear_corpus = []  # corpus as sequence of words, prepared for models not requiring syntax
        self.p_noun = []  # patient nouns
        self.t_verb = []  # transitive verbs
        self.t_p_pairs = {}  # verb_patient pairs
        self.verb = [] # verbs
        self.agent = [] # agents
        self.v_a_pairs = {} # vern_agent pairs
        self.noun_stems = [] # noun wthout thematic roles
        self.noun_dict = {} # dictionary for noun stems and their roles

        # action_dict = {'hunt_deer': [A, B, C, D]}

    def get_activated_words(self):
        l = len(self.corpus)
        steve = STN.Stn(self.corpus)
        # animal = set(the_world.animal_category)
        hunting = {'trapping'}
        if len(hunting.intersection(set(steve.word_list))) == 0:
            words = random.choices(steve.word_list, k=1)
        else:
            hunting_list = list(set(steve.word_list).intersection(hunting))
            words = random.choices(hunting_list, k=1)
        return words,steve

    ################################################################################################################
    # get the things that will be searched for food
    ################################################################################################################

    def get_target(self):
        target_list = []
        for herbivore in self.world.food_herbivore_list:
            if herbivore not in self.world.searched_list:
                target_list.append(herbivore)
        for nut in self.world.nut_list:
            target_list.append(nut)
        for fruit in self.world.fruit_list:
            target_list.append(fruit)
        return target_list

    ################################################################################################################
    # get hunting method from success rates for all hunting actions
    ################################################################################################################
    def get_hunting_method(self):
        t = self.event_tree
        hunting_skill, length = self.get_hunting_skill()
        hunting_method = {}
        method_list = [(0,0,0,2,0,0), (0,0,0,2,0,1), (0,0,0,2,0,2)]
        for method in method_list:
            skills = [n for n in t.neighbors(method)]
            success_rate = np.ones(length)
            for skill in skills:
                skill_name = self.event_dict[skill][0]
                skill_rate = hunting_skill[skill_name]
                success_rate = np.multiply(success_rate, skill_rate)
            hunting_method[method] = success_rate
        return hunting_method

    @staticmethod
    ################################################################################################################
    # get hunting method from success rates for all hunting actions, which is recorded in the random_sampling file
    # under the directory
    ################################################################################################################
    def get_hunting_skill():
        hunting_skill = {}
        with open('World/human_hunt.csv') as csv_file:
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
    ################################################################################################################
    # get the energy needed for all actions
    ################################################################################################################
    def get_state_change():
        state_change = {}
        with open('World/state_change.csv') as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            line = 0
            for row in csv_reader:
                if line > 0:
                    copy = []
                    for item in row[1:]:
                        copy.append(eval(item))
                    state_change[row[0]] = np.asarray(copy)
                line = line + 1
        # print(state_change)
        return state_change

    ####################################################################################################################
    # taking turn function for human
    # humans act due to the event tree structure
    # at every moment, a human is on some node of the structure, if on a terminal node, carrying out the corresponding
    # simple event, when a simple event is carried out, humans are actually doing something, as a result, the level of
    # drives got updated, and the changing rate for eahc simple event is recorded in the state_change file under the
    # directory

    # otherwise, decide which node to go for the next moment and move to that node, which are 'inner decisions'
    # therefore do not effect on the level of drives
    ####################################################################################################################
    def take_turn(self):
        t = self.event_tree
        self.compute_status()  # no matter which node the human is on at the moment, compute the status of that
        # node(event).
        status = self.event_dict[self.current_event][1]
        #print('{}{} on status {}'.format(self.name, self.id_number, self.current_event))
        #print(self.food_stored, self.food_list, self.focus)
        #print(self.hunger, self.sleepiness, self.thirst)


        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # status 1 means event not completed, keep carrying out functions for simple event
                event_name = self.event_dict[self.current_event][0]
                if event_name == 'searching':
                    self.searching()
                elif event_name == 'going_to':
                    self.going_to()
                elif event_name in {'gathering', 'butchering', 'cooking', 'eating', 'laying_down', 'sleeping',
                                    'waking_up','yawning','stretching','getting_up','drinking','resting','peeling','cracking'}:
                    self.do_it(event_name)
                else:
                    self.hunt(event_name)
                self.generate_language(event_name)  # generate the corresponding sentence for the ongoing simple event
            elif status == 0: # status 0 means event completed
                self.current_event = self.current_event[:len(self.current_event)-1]  # go back to the parent node
            else:  # status -1 means event failed, re-initialize the event tree and restart
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_human, 0)
                self.focus = None
                self.current_event = ()

        elif t.in_degree(self.current_event) != 0:  # currently on branch
            if status == 0:  # event completed, go back to parent node
                self.current_event = self.current_event[:len(self.current_event)-1]
            else:  # event not completed, decide which child to go and go to that child
                self.current_event = self.choose_heir()

        else:  # currently on the root
            if status == 0:  # event tree have been completed, re-initialize the event tree
                if VERBOSE:
                    print('for {}, epoch finished'.format(self.name))
                    print(self.hunger, self.sleepiness, self.thirst)
                l = len(self.world.human_list)
                if self.world.human_list.index(self) == l-1:
                    self.world.epoch = self.world.epoch + 1
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_human,0)
                self.focus = None
                self.food_stored = self.food_stored/2
                for food in self.food_list:
                    food.remain_size = food.remain_size/2 # food rote
            else:  # event tree not completed, choose the child node to go and go to the child.
                self.current_event = self.choose_heir()

        #  print(self.eat_count)
        #  print(self.sleep_count)
        #  print(self.drink_count)
        #  print(self.idle_count)

    ####################################################################################################################
    #  generating the language(corpus) as the description of the events that the humans witnessed
    ####################################################################################################################
    def generate_language(self,event_name):
        agent = self.name + '-a'
        if agent not in self.agent:
            self.agent.append(agent)
        if agent not in self.world.agent:
            self.world.agent.append(agent)
        if event_name not in self.verb:
            self.verb.append(event_name)
        if event_name not in self.world.verb:
            self.world.verb.append(event_name)

        #  if self.current_event[0] == 0:
        #      print('{} is hungry.'.format(self.name))
        #  elif self.current_event[0] == 1:
        #      print('{} is sleepy.'.format(self.name))
        #  elif self.current_event[0] == 2:
        #      print('{} is thirsty.'.format(self.name))

        # collect noun stems:
        the_noun = self.name
        if the_noun not in self.noun_stems:
            self.noun_dict[the_noun] = [agent]
            self.noun_stems.append(the_noun)
        else:
            if agent not in self.noun_dict[the_noun]:
                self.noun_dict[the_noun].append(agent)
                self.noun_dict[the_noun].sort()

        if the_noun not in self.world.noun_stems:
            self.world.noun_dict[the_noun] = [agent]
            self.world.noun_stems.append(the_noun)
        else:
            if agent not in self.world.noun_dict[the_noun]:
                self.world.noun_dict[the_noun].append(agent)
                self.world.noun_dict[the_noun].sort()

        focus = None
        the_patient = None
        if  event_name == 'searching' or event_name == 'resting' or event_name != 'going_to' and \
            self.focus is None :
            if VERBOSE:
                print('{} is {}.'.format(agent, event_name))

            sentence = (agent, event_name)
            linear_sent = [agent, event_name, '.']

        else:
            if event_name == 'going_to':
                the_patient = self.destination
                focus = self.destination + '-p'
                if VERBOSE:
                    print('{} is {} {}.'.format(agent, event_name, focus))

            elif isinstance(self.focus, animals.Herbivore): # when patient is animate
                the_patient = self.focus.category
                focus = self.focus.category + '-p'
                if VERBOSE:
                    print('{} is {} the {}.'.format(agent, event_name, focus))

            else:
                if type(self.focus) is type('1'): # when patient is mass
                    the_patient = self.focus
                    focus = self.focus + '-p'
                else: # when patient is inanimate object (fruits, nuts, meat)
                    the_patient = self.focus.category
                    focus = self.focus.category + '-p'
                if VERBOSE:
                    print('{} is {} {}.'.format(agent, event_name, focus))

            sentence = (agent,(event_name,focus))
            linear_sent = [agent, event_name, focus, '.']
        if self.corpus == [] or self.corpus[-1] != sentence:
            self.corpus.append(sentence)
            self.world.corpus.append(sentence)
            self.linear_corpus.append(linear_sent)
            self.world.linear_corpus.append(linear_sent)

            if (event_name, agent) not in self.v_a_pairs:  # record verb_agent
                self.v_a_pairs[(event_name, agent)] = 1
            else:
                self.v_a_pairs[(event_name, agent)] += 1
            if (event_name, agent) not in self.world.v_a_pairs:
                self.world.v_a_pairs[(event_name, agent)] = 1
            else:
                self.world.v_a_pairs[(event_name, agent)] += 1

            if focus != None:
                if focus not in self.p_noun:
                    self.p_noun.append(focus)
                if focus not in self.world.p_noun:
                    self.world.p_noun.append(focus)
                if event_name not in self.t_verb:
                    self.t_verb.append(event_name)
                if event_name not in self.world.t_verb:
                    self.world.t_verb.append(event_name)
                if (event_name,focus) not in self.t_p_pairs:  # record all verb phrases (transitive events) occurred
                    self.t_p_pairs[(event_name,focus)] = 1
                else:
                    self.t_p_pairs[(event_name,focus)] += 1
                if (event_name, focus) not in self.world.t_p_pairs:
                    self.world.t_p_pairs[(event_name,focus)] = 1
                else:
                    self.world.t_p_pairs[(event_name, focus)] += 1

                # collect noun stems:
                if the_patient not in self.noun_stems:
                    self.noun_dict[the_patient] = [focus]
                    self.noun_stems.append(the_patient)
                else:
                    if focus not in self.noun_dict[the_patient]:
                        self.noun_dict[the_patient].append(focus)
                        self.noun_dict[the_patient].sort()

                if the_patient not in self.world.noun_stems:
                    self.world.noun_dict[the_patient] = [focus]
                    self.world.noun_stems.append(the_patient)
                else:
                    if focus not in self.world.noun_dict[the_patient]:
                        self.world.noun_dict[the_patient].append(focus)
                        self.world.noun_dict[the_patient].sort()

    ####################################################################################################################
    #  when on a non-terminal node and the event corresponding to the node has not been completed, it decides which
    #  child node it goes

    #  all non-terminal nodes are categorized into serial, p-parallel or o-parallel, where the children of serial nodes
    #  are aligned in order from left to right, so that the children are completed from left to right, therefore, for a
    #  serial node, the choice is deterministic, it is the leftmost child which has not be completed yet

    # for parallel nodes, the pp nodes choose the child by some random sampling, while the op nodes choose the child
    # by first computing the score of each child, and go to the child with the highest score
    ####################################################################################################################
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
            if self.current_event == (2,):  # choose the drink to drink
                self.focus = random.choice(self.world.drink_category)

        else:
            if self.current_event == (0,0,0,2,0):  # choose hunting method
                hunting_method_dist = []
                index = self.world.herbivore_category.index(self.focus.category)
                for child in children:
                    hunting_method_dist.append(self.hunting_method[child][index])
                norm = [float(i) / sum(hunting_method_dist) for i in hunting_method_dist]
                self.current_event = children[int(np.random.choice(len(children), 1, p=norm)[0])]
        return self.current_event

    def compute_scores(self, event):
        if self.current_event == (): # decide what to satisfy
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
        elif self.current_event == (0, 0): # decide whether go out for food
            if event == (0, 0, 0):
                score = self.hunger
            else:
                score = self.food_stored

        elif self.current_event == (0,1,1): # decide whether cook or peel or crash
            if self.focus == None:
                self.focus = self.food_list[0]

            if event == (0,1,1,0): # animal: cook
                if self.focus.category in self.world.herbivore_category:
                    score = 1
                else:
                    score = 0
            elif event == (0,1,1,1): # nut: crack
                if self.focus.category in self.world.nut_category:
                    score = 1
                else:
                    score = 0
            else: # fruit: peel
                if self.focus.category in self.world.fruit_category:
                    score = 1
                else:
                    score = 0

        elif self.current_event == (0,0,0,2): # decide whether hunt or gather
            if event == (0,0,0,2,0):
                if self.focus.category in self.world.herbivore_category: # find herbivore, go hunt
                    score = 1
                else:
                    score = 0
            else:
                if self.focus.category not in self.world.herbivore_category: # find fruit or nut, go gather
                    score = 1
                else:
                    score = 0

        else: # decide whether butcher
            if event == (0, 0, 0, 4, 0):
                if self.focus.category in self.world.herbivore_category:  # find herbivore, go hunt
                    score = 1
                else:
                    score = 0
            else:
                if self.focus.category not in self.world.herbivore_category:  # find fruit or nut, go gather
                    score = 1
                else:
                    score = 0
        return score

    ####################################################################################################################
    # computing the status of the node standing on at the moment
    # the status of leaves is the status of itself, 0,1, or -1
    # the status of non terminal nodes are functions of the status of their children, for serial nodes, it is the
    # Boolean product of children's status, and for parallel nodes, it is the Boolean sum of children's status
    ####################################################################################################################
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

    ####################################################################################################################
    # function for simple event 'searching', look for the animal that is closest to the human, and focus on the animal
    # the event is completed once an animal is found (focused)
    ####################################################################################################################
    def searching(self):
        region = random.choice(['east', 'west', 'north', 'south'])
        game_list = []
        original_position = (self.x, self.y)
        self.food_target = self.get_target()
        if region == 'east':
            for target in self.food_target:
                if target.x > abs(target.y):
                    game_list.append(target)
            self.x = config.World.world_size/4
            self.y = 0
        elif region == 'west':
            for target in self.food_target:
                if -target.x > abs(target.y):
                    game_list.append(target)
            self.x = -config.World.world_size / 4
            self.y = 0
        elif region == 'north':
            for target in self.food_target:
                if target.y > abs(target.x):
                    game_list.append(target)
            self.y = config.World.world_size / 4
            self.x = 0
        else:
            for target in self.food_target:
                if -target.y > abs(target.x):
                    game_list.append(target)
            self.y = -config.World.world_size / 4
            self.x = 0
        movement = ((self.x - original_position[0])**2 + (self.y - original_position[1])**2)**0.5
        if len(game_list) > 0:
            choice = random.randint(0,len(game_list)-1)
            self.focus = game_list[choice]
            if self.focus.type == 'animal': # if found an herbivore animal
                self.world.searched_list.append(self.focus)
            self.event_dict[self.current_event][1] = 0
        self.hunger = self.hunger + movement * self.state_change['searching'][0]

    ####################################################################################################################
    # function for simple event 'going_to'
    # if the focus is an animal move toward the animal with the person's speed, and the event is completed when the
    # animal is in vision

    # if the focus is not an animal, the event is completed
    ####################################################################################################################
    def going_to(self):
        if self.current_event == (0,0,0,0,1):
            self.destination = self.focus.category
            d = ((self.x - self.focus.x)**2 + (self.y - self.focus.y)**2)**0.5
            if d <= self.vision:
                self.event_dict[self.current_event][1] = 0
            else:
                dx = self.x - self.focus.x
                dy = self.y - self.focus.y
                norm = (dx ** 2 + dy ** 2) ** 0.5
                dx = dx / norm
                dy = dy / norm
                if d > self.vision + self.speed:
                    self.x = self.x - dx * self.speed
                    self.y = self.y - dy * self.speed
                    self.hunger = self.hunger + self.speed * self.state_change['going_to'][0]
                else:
                    self.x = self.focus.x
                    self.y = self.focus.y
                    self.hunger = self.hunger + d * self.state_change['going_to'][0]
        else:
            if self.current_event == (0, 1, 0):  # go for eat
                self.destination = 'fire'
            elif self.current_event == (1, 0):  # go sleeping
                self.destination = 'tent'
            else:  # go get drinks
                self.destination = 'river'
            self.event_dict[self.current_event][1] = 0

    ####################################################################################################################
    # function for hunting events
    # each hunting event is a serial combination of two simple events
    # at hunting node, choose the hunting method(event) according to the hunting method distribution corresponding to
    # the focused animal. If focusing on a rabbit, choose the hunting method according to the method distribution
    # regarding to rabbit.

    # Method chosen, conduct the hunting series, each action of the series might fail. If it fails, it means that the
    # the animal is not caught and need to start over from the root on the event tree
    ####################################################################################################################

    def hunt(self,event_name):
        hunting_skill = self.get_hunting_skill()[0]
        index = self.world.herbivore_category.index(self.focus.category)
        success_rate = hunting_skill[event_name][index]
        num = np.random.choice(2, 1, p=[1-success_rate, success_rate])[0]
        if num == 1:
            self.event_dict[self.current_event][1] = 0
            exception = {'chasing','trapping','catching'}
            if event_name not in exception:
                if self.focus in self.world.herbivore_list:
                    self.world.herbivore_list.remove(self.focus)
                    self.focus.alive = False
                self.food_target = self.food_target.remove(self.focus)
        else:
            self.event_dict[self.current_event][1] = -1
            self.world.searched_list.remove(self.focus)
            if VERBOSE:
                print('hunting action failed')
        self.hunger = self.hunger + self.state_change[event_name][0]

    ####################################################################################################################
    # functions for all other simple events, some of which are specific, requiring specific functions, other are more
    # general, only result in changes in drives
    ####################################################################################################################
    def do_it(self,event_name):

        if event_name == 'gathering':  # gather the animal or other objects,  hunger increase
            # proportional to the size of the animal or the objects (fruit, nut), otherwise, hunger stay the same
            if type(self.focus) is not type('1') :
                self.hunger = self.hunger + self.focus.size * self.state_change[event_name][0]
                if self.focus.type is 'plant_r':
                    self.focus.gather_size = random.uniform(0.1,0.9) * self.focus.size
                    if self.focus.gather_size >= self.focus.remain_size:
                        self.focus.remain_size = 0
                    else:
                        self.focus.remain_size = self.focus.remain_size - self.focus.gather_size
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'butchering':  # once butchered, the animal disappears and turned into food, and get stored
            # the size of the food is half of the size of the animal, and the hunger increases proportional to the size
            # of the animal
            if self.focus.type == 'animal':
                self.focus.remain_size = self.focus.size/2
                self.food_list.append(self.focus)
                self.food_stored = self.food_stored + self.focus.remain_size
                self.hunger = self.hunger + self.state_change[event_name][0] * self.focus.remain_size
                self.sleepiness = self.sleepiness + self.state_change[event_name][1]
                self.hunger = self.hunger + self.focus.size * self.state_change[event_name][0]
                self.event_dict[self.current_event][1] = 0
            else:
                self.event_dict[self.current_event][1] = 0

        elif event_name == 'cooking':  # when cooking, the focus becomes the food in the storage. Cook the amount of
            # food which satisfies current hunger when there is enough food, otherwise cook all food
            # once a piece of food is cooked, the food is turned into a dish
            # cooking is completed once all food stored have been cooked, or the amount of food cooked satisfies the
            # current hunger
            amount_need = self.hunger - self.dish_amount
            self.focus = self.food_list[0]
            if self.focus.remain_size >= amount_need:
                self.focus.remain_size = self.focus.remain_size - amount_need
                self.food_stored = self.food_stored - amount_need
                self.dish_amount = self.hunger
                self.event_dict[self.current_event][1] = 0

            else:
                self.dish_amount = self.dish_amount + self.focus.remain_size
                self.food_stored = self.food_stored - self.focus.remain_size
                self.food_list.remove(self.focus)
                self.world.consumption.append((self.focus.category,self.focus.size/2))
                if len(self.food_list) == 0:
                    self.event_dict[self.current_event][1] = 0

            self.dish_list.append(self.focus.category)
            if self.event_dict[self.current_event][1] == 0:
                dish_set = set(self.dish_list)
                self.dish_list = list(dish_set)

        elif event_name == 'eating':
            self.world.human_eat += 1
            if  self.focus in self.world.herbivore_category or self.focus.type == 'animal':  # eat the dish one by one, completed when all dishes are gone
                self.focus = self.dish_list.pop()
                self.eat_count_meal = self.eat_count_meal + 1
                if len(self.dish_list) == 0:
                    self.hunger = self.hunger - self.dish_amount
                    self.sleepiness = self.sleepiness + self.sleepy_rate * self.dish_amount
                    self.dish_amount = 0
                    self.event_dict[self.current_event][1] = 0
            else:  # eat fruits, or nuts
                if self.focus.category in self.world.fruit_category:
                    self.eat_count_fruit = self.eat_count_fruit + 1
                else:
                    self.eat_count_nut = self.eat_count_nut + 1
                if self.focus.gather_size >= self.hunger:
                    self.sleepiness = self.sleepiness + self.sleepy_rate * self.hunger
                    self.hunger = 0
                else:
                    self.sleepiness = self.sleepiness + self.sleepy_rate * self.focus.gather_size
                    self.hunger = self.hunger - self.focus.gather_size
                self.event_dict[self.current_event][1] = 0

        elif event_name == 'sleeping':  # sleep completed immediately
            self.sleep_count = self.sleep_count + 1
            self.sleepiness = 0
            self.event_dict[self.current_event][1] = 0

        elif event_name == 'drinking':  # drinking completed immediately
            self.drink_count = self.drink_count + 1
            # print(self.thirst)
            self.thirst = 0
            self.event_dict[self.current_event][1] = 0

        else:
            if event_name == 'resting':  # when have nothing to do, drive still get updated
                self.idle_count = self.idle_count + 1

            #  for events not specified, just update the drives
            self.hunger = self.hunger + self.state_change[event_name][0]
            self.event_dict[self.current_event][1] = 0

        self.sleepiness = self.sleepiness + self.state_change[event_name][1]
        self.thirst = self.thirst + self.state_change[event_name][2]




