import random
import csv
import numpy as np

from Programs.World import config
from Programs.World import event_tree as et
from Programs.World import human
VERBOSE = False

class Agent:
    ####################################################################################################################
    # Agents are entities in the world that is animate and be able to play agent role in events
    ####################################################################################################################
    def __init__(self, world):
        self.world = world
        self.x = None
        self.y = None
        self.type = None
        self.id_number = None

        # a human has basic drives which motivate their actions, currently including hunger, sleepiness and thirst
        self.hunger = None
        self.sleepiness = None
        self.thirst = None

        # threshold for drives any abilities
        self.food_threshold = None
        self.sleep_threshold = None
        self.hunger_threshold = None
        self.thirst_threshold = None

        # the rate for getting sleepy while idling
        self.sleepy_rate = None

        # basic physical attributes, vision is the range for detecting an animal
        self.speed = None
        self.vision = None

        # other attributes which may be included in more complex world designs
        self.fatigue = None
        self.energy = None
        self.health = None
        self.max_speed = None
        self.awake = None
        self.age = None
        self.size = None
        self.strength = None
        self.intelligence = None

        ################################################################################################################
        # event related attributes
        ################################################################################################################

        self.focus = None  # when human are involved in transitive(ditransitive) events, focus is the patient, otherwise
        # it is none
        self.event_dict = None
        self.event_tree = None
        self.destination = None  # where to go
        self.food_target = None  # get the food target
        self.state_change = None  # get the change of drive levels for all actions.
        self.drive = ['hunger', 'sleepiness', 'thirst']
        self.current_event = ()  # where the agent is on the event tree at the moment

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
        self.v_a_pairs = {} # verb_agent pairs
        self.noun_stems = [] # noun wthout thematic roles
        self.noun_dict = {} # dictionary for noun stems and their roles

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
            pass
        return self.current_event

    def compute_scores(self, event): # as an agent, it has to satisfy hunger, thirst and sleepiness
        # decide what to satisfy
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
        return score


class Animal(Agent):

    ####################################################################################################################
    # define animal class
    # animal object has category, size and an randomized initial location
    # speed is a constant rate that the animal moves
    # vision is a range that within which the animal detect a human's presence
    # For current experiment designs, deer and rabbit are the only two categories of animals may be generated
    ####################################################################################################################

    def __init__(self, world, category):
        Agent.__init__(self,world)
        self.x = random.randint(-config.World.world_size/5, config.World.world_size/5)
        self.y = random.randint(-config.World.world_size/5, config.World.world_size/5)
        self.type = 'animal'
        self.category = category
        self.state_change = self.get_state_change()
        self.food_target = self.get_target()
        self.eat_count_meal = 0
        self.eat_count_nut = 0
        self.eat_count_fruit = 0
        self.eat_count_plant = 0
        self.drink_count = 0
        self.sleep_count = 0
        self.idle_count = 0
        self.alive = True
        self.animal_type = None


    ################################################################################################################
    # get the things that will be searched for food
    ################################################################################################################

    def get_target(self):
        target_list = []
        return target_list

    ################################################################################################################
    # get the energy needed for all actions
    ################################################################################################################
    @staticmethod
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
    #  generating the language(corpus) as the description of the events that the animals witnessed
    ####################################################################################################################
    def generate_language(self, event_name):
        #  if self.current_event[0] == 0:
        #      print('{} is hungry.'.format(self.name))
        #  elif self.current_event[0] == 1:
        #      print('{} is sleepy.'.format(self.name))
        #  elif self.current_event[0] == 2:
        #      print('{} is thirsty.'.format(self.name))

        # collect agents and agent-verb
        agent = self.category + '-a'
        the_noun = self.category
        if agent not in self.agent:
            self.agent.append(agent)
        if agent not in self.world.agent:
            self.world.agent.append(agent)
        if event_name not in self.verb:
            self.verb.append(event_name)
        if event_name not in self.world.verb:
            self.world.verb.append(event_name)


        # collect noun stems:
        if the_noun not in self.noun_stems:
            self.noun_dict[the_noun]=[agent]
            self.noun_stems.append(the_noun)
        else:
            if agent not in self.noun_dict[the_noun]:
                self.noun_dict[the_noun].append(agent)
                self.noun_dict[the_noun].sort()

        if the_noun not in self.world.noun_stems:
            self.world.noun_dict[the_noun]=[agent]
            self.world.noun_stems.append(the_noun)
        else:
            if agent not in self.world.noun_dict[the_noun]:
                self.world.noun_dict[the_noun].append(agent)
                self.world.noun_dict[the_noun].sort()

        focus = None
        the_patient = None
        if event_name == 'searching' or event_name == 'resting' or event_name != 'going_to' and self.focus is None:
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

            elif isinstance(self.focus, Herbivore):  # when patient is a herbivore
                the_patient = self.focus.category
                focus = self.focus.category + '-p'
                if VERBOSE:
                    print('{} is {} the {}.'.format(agent, event_name, focus))

            elif isinstance(self.focus, human.Human):  # when patient is a human
                the_patient = self.focus.name
                focus = self.focus.name + '-p'
                if VERBOSE:
                    print('{} is {} the {}.'.format(agent, event_name, focus))

            else:
                if type(self.focus) is type('1'):  # when patient is mass
                    the_patient = self.focus
                    focus = self.focus + '-p'
                else:  # when patient is inanimate object (fruits, nuts, meat)
                    the_patient = self.focus.category
                    focus = self.focus.category + '-p'
                if VERBOSE:
                    print('{} is {} {}.'.format(agent, event_name, focus))

            sentence = (agent, (event_name, focus))
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
                # collect patient nouns and verb-patient pairs
                if focus not in self.p_noun:
                    self.p_noun.append(focus)
                if focus not in self.world.p_noun:
                    self.world.p_noun.append(focus)
                if event_name not in self.t_verb:
                    self.t_verb.append(event_name)
                if event_name not in self.world.t_verb:
                    self.world.t_verb.append(event_name)
                if (event_name, focus) not in self.t_p_pairs:  # record all verb phrases (transitive events) occurred
                    self.t_p_pairs[(event_name, focus)] = 1
                else:
                    self.t_p_pairs[(event_name, focus)] += 1
                if (event_name, focus) not in self.world.t_p_pairs:
                    self.world.t_p_pairs[(event_name, focus)] = 1
                else:
                    self.world.t_p_pairs[(event_name, focus)] += 1

                # collect noun stems
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
            if self.focus.type == 'human' or self.focus.type == 'animal': # found a human or an herbivore animal
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
        if type(self.focus) is type('1') :  # whent the animal wants to drink
            self.destination = 'river'
            self.event_dict[self.current_event][1] = 0

        elif self.focus is not None : # when the animal have found something
            if self.focus.type == 'human':
                self.destination = self.focus.name
            else:
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



    def do_it(self,event_name):

        if event_name == 'eating': # when an animal eat something, it eats until it is full when there is enough food,
            # otherwise, it eats up all the food.
            if self.animal_type == 'herbivore':
                self.world.herbivore_eat += 1
            else:
                self.world.carnivore_eat += 1

            if self.focus.type is 'animal' or self.focus in self.world.herbivore_category:
                self.eat_count_meal = self.eat_count_meal + 1
            if self.focus.category in self.world.plant_category:
                self.eat_count_plant = self.eat_count_plant + 1

            if self.focus.remain_size >= self.hunger:
                self.sleepiness = self.sleepiness + self.sleepy_rate * self.hunger
                self.focus.remain_size = self.focus.remain_size - self.hunger
                self.hunger = 0
            else:
                self.sleepiness = self.sleepiness + self.sleepy_rate * self.focus.remain_size
                self.hunger = self.hunger - self.focus.remain_size
                self.focus.remain_size = 0
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




class Carnivore(Animal):
    def __init__(self, world, category):
        Animal.__init__(self, world, category)
        self.id_number = len(self.world.carnivore_list)
        self.animal_type = 'carnivore'


        self.hunger = random.uniform(0.5,0.7)
        self.sleepiness = random.uniform(0,0.5)
        self.thirst = random.uniform(0.7,1)

        self.sleep_threshold = 1
        self.hunger_threshold = 100
        self.thirst_threshold = 0.3
        self.sleepy_rate = 0.05

        self.speed = random.uniform(200, 300)
        self.vision = random.uniform(80, 120)
        self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_carnivore, 0)  # generate
        # the event tree structure which human obeys.
        # animal size refers to the weight of the animal object, which is transfered to many objects in the world, when
        # animals are hunted and butchered, the animal objects are turned into certain amount of food, the amount of
        # food is determined by the size of the animal.
        self.size = random.randint(self.world.animal_size[self.category][0], self.world.animal_size[self.category][1])
        self.remain_size = self.size

    def get_target(self):
        target_list = []
        for herbivore in self.world.herbivore_list:
            if herbivore not in self.world.searched_list:
                target_list.append(herbivore)
        for human in self.world.human_list:
            if human not in self.world.searched_list:
                target_list.append(human)
        return target_list

    def take_turn(self):
        t = self.event_tree
        self.compute_status()  # no matter which node the human is on at the moment, compute the status of that
        # node(event).
        status = self.event_dict[self.current_event][1]
        # print('{}{} on status {}'.format(self.name, self.id_number, self.current_event))
        # print(self.hunger, self.sleepiness, self.thirst)


        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # status 1 means event not completed, keep carrying out functions for simple event
                event_name = self.event_dict[self.current_event][0]
                if event_name == 'searching':
                    self.searching()
                elif event_name == 'going_to':
                    self.going_to()
                elif event_name in {'gathering', 'butchering', 'cooking', 'eating', 'laying_down', 'sleeping',
                                    'waking_up', 'getting_up','drinking','resting','peeling','cracking'}:
                    self.do_it(event_name)
                else:
                    self.hunt(event_name)
                self.generate_language(event_name)  # generate the corresponding sentence for the ongoing simple event
            elif status == 0: # status 0 means event completed
                self.current_event = self.current_event[:len(self.current_event)-1]  # go back to the parent node
            else:  # status -1 means event failed, re-initialize the event tree and restart
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_carnivore, 0)
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
                    print('epoch finished')
                    print(self.hunger, self.sleepiness, self.thirst)
                l = len(self.world.carnivore_list)
                if self.world.carnivore_list.index(self) == l-1:
                    self.world.epoch = self.world.epoch + 1
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_carnivore,0)
                self.focus = None
            else:  # event tree not completed, choose the child node to go and go to the child.
                self.current_event = self.choose_heir()



    ################################################################################################################
    # get hunting method from success rates for all hunting actions, which is recorded in the random_sampling file
    # under the directory
    ################################################################################################################
    @staticmethod
    def get_hunting_skill():
        hunting_skill = {}
        with open('World/carnivore_hunt.csv') as csv_file:
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

    def hunt(self,event_name):
        hunting_skill = self.get_hunting_skill()[0]
        if self.focus.type == 'animal': # when hunting a herbivore
            index = self.world.herbivore_category.index(self.focus.category)
        else: # when hunting a human
            index = len(self.world.herbivore_category)
        success_rate = hunting_skill[event_name][index]
        num = np.random.choice(2, 1, p=[1-success_rate, success_rate])
        if num == 1:
            self.event_dict[self.current_event][1] = 0
            exception = {'chasing'}
            if event_name not in exception:
                if self.focus in self.world.herbivore_list:
                    self.world.herbivore_list.remove(self.focus)
                    self.focus.alive = False
                elif self.focus in self.world.human_list:
                    self.world.human_list.remove(self.focus)
                    self.focus.alive = False
                self.food_target.remove(self.focus)
        else:
            self.event_dict[self.current_event][1] = -1
            self.world.searched_list.remove(self.focus)
            if VERBOSE:
                print('hunting action failed')
        self.hunger = self.hunger + self.state_change[event_name][0]





class Herbivore(Animal):
    def __init__(self, world, category):
        Animal.__init__(self, world, category)
        self.id_number = len(self.world.herbivore_list)
        self.animal_type = 'herbivore'

        self.hunger = random.uniform(0.5,0.7)
        self.sleepiness = random.uniform(0,0.5)
        self.thirst = random.uniform(0.7,1)

        self.sleep_threshold = 1
        self.hunger_threshold = 1
        self.thirst_threshold = 0.3
        self.sleepy_rate = 0.05


        self.speed = random.uniform(1, 5)
        self.vision = random.uniform(30, 60)
        self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_herbivore, 0)  # generate
        # the event tree structure which human obeys.
        # animal size refers to the weight of the animal object, which is transfered to many objects in the world, when
        # animals are hunted and butchered, the animal objects are turned into certain amount of food, the amount of
        # food is determined by the size of the animal.
        self.size = random.randint(self.world.animal_size[self.category][0], self.world.animal_size[self.category][1])
        self.remain_size = self.size

    def get_target(self):
        target_list = []
        for plant in self.world.plant_list:
            target_list.append(plant)
        return target_list

    ####################################################################################################################
    # define take turn function for a herbivore.
    # A herbivore runs if it detect a human or carnivore, stays at where it is otherwise
    ####################################################################################################################

    def take_turn(self):
        t = self.event_tree
        self.compute_status()  # no matter which node the human is on at the moment, compute the status of that
        # node(event).
        status = self.event_dict[self.current_event][1]
        # print('{}{} on status {}'.format(self.name, self.id_number, self.current_event))
        # print(self.hunger, self.sleepiness, self.thirst)

        if t.out_degree(self.current_event) == 0:  # currently on leave
            if status == 1:  # status 1 means event not completed, keep carrying out functions for simple event
                event_name = self.event_dict[self.current_event][0]
                if event_name == 'searching':
                    self.searching()
                elif event_name == 'going_to':
                    self.going_to()
                elif event_name in {'gathering', 'butchering', 'cooking', 'eating', 'laying_down', 'sleeping',
                                    'waking_up', 'getting_up', 'drinking', 'resting', 'peeling', 'cracking'}:
                    self.do_it(event_name)
                self.generate_language(event_name) # generate the corresponding sentence for the ongoing simple event
            else :  # status 0 means event completed
                self.current_event = self.current_event[:len(self.current_event) - 1]  # go back to the parent node

        elif t.in_degree(self.current_event) != 0:  # currently on branch
            if status == 0:  # event completed, go back to parent node
                self.current_event = self.current_event[:len(self.current_event) - 1]
            else:  # event not completed, decide which child to go and go to that child
                self.current_event = self.choose_heir()

        else:  # currently on the root
            if status == 0:  # event tree have been completed, re-initialize the event tree
                if VERBOSE:
                    print('epoch finished')
                    print(self.hunger, self.sleepiness, self.thirst)
                l = len(self.world.herbivore_list)
                if self.world.herbivore_list.index(self) == l - 1:
                    self.world.epoch = self.world.epoch + 1
                self.event_dict, self.event_tree = et.initialize_event_tree(config.World.event_tree_herbivore, 0)
                self.focus = None
            else:  # event tree not completed, choose the child node to go and go to the child.
                self.current_event = self.choose_heir()


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
            pass
        return self.current_event

    def escape(self):
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
    # define run function for an herbivores.
    # An animal runs away from the human it detect, that is, the opposite direction of which facing the human
    ####################################################################################################################

    def run(self, human, d):
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



