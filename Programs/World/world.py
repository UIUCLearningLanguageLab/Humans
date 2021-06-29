import random
from Programs.World import config
from Programs.World.human import Human

import Programs.World.animals as animal
import Programs.World.plant_rescource as plant


class World:
    ################################################################################################################
    # define the world object, detailed setting of the world
    # the world is first generated, and then humans and animals
    # the world generated serves as a required attribute for the humans and animals, it indicate which world is the
    # human and the animal living in

    # all agents in the world conform to the world principals listed below
    ################################################################################################################

    def __init__(self):

        #  type of the predicates(simple event), the number refers to the argument it takes,
        self.simple_event_dict = {'search': 1, 'go_to': 2, 'trap': 2, 'catch': 2, 'chase': 2, 'stab': 2, 'shoot': 2,
                                  'throw_at': 2, 'gather': 2, 'butcher': 2, 'cook': 2, 'eat': 2, 'lay_down': 1,
                                  'asleep': 1, 'wake_up': 1, 'get_up': 1}


        ###########################################################################################################
        #Animate
        ###########################################################################################################

        self.agent_list = []

        # possible names of the humans in the world, which are the names of lab members in Learning & Language Lab
        self.name_list = ['Jessica','Jon','Anastasia','Phil','Andrew','Lin Khern','Emily','Shufan','Jacki','Katherine']
        self.human_list = []

        # animal taxonomy
        self.animal_taxo = {'herbivore':['rabbit', 'squirrel', 'fox', 'mouflon', 'boar', 'ibex', 'bison',
                           'buffalo','auroch'],'carnivore':['wolf','tiger','hyena']}

        self.animal_type = ['carnivore','herbivore']

        self.herbivore_category = ['rabbit', 'squirrel', 'boar', 'ibex', 'bison',
                                   'buffalo','auroch','fox', 'mouflon',]
        self.carnivore_category = ['wolf','tiger','hyena']

        # the range of size of an animal object in each category
        self.animal_size = {'rabbit':(3,5), 'squirrel':(1,2), 'fox':(9,19), 'mouflon':(50,100),
                            'boar':(88,110), 'ibex':(80,150), 'bison':(1800,2200), 'buffalo': (1300,2600),
                                'auroch':(1000,2000), 'tiger':(400,600),'wolf':(70,100),'hyena':(90,120)}
        self.herbivore_list = []
        self.carnivore_list = []

        # animals already been searched, not longer in the pool for further searching
        self.searched_list = []

        ###########################################################################################################
        # plant_resources
        ###########################################################################################################

        # plant taxonomy:
        self.plant_taxo = {'fruit':['apple','peach','pear'],'nut':['walnut','cashew','almond'],'plant':['leaf','grass',
                                                                                                        'flower']}
        self.plant_type = ['fruit','nut','plant']

        # possible fruit categories
        self.fruit_category = ['apple','peach','pear']

        # possible nut categories
        self.nut_category = ['walnut','cashew','almond']

        # possible plant categories
        self.plant_category = ['leaf','grass','flower']

        # plant grows
        self.grow_rate = {'apple':0.1,'peach':0.1,'pear':0.1, 'walnut':0.3, 'cashew':0.3, 'almond':0.3, 'leaf':0.5,
                          'grass':0.5, 'flower':0.5}
        self.plant_resource = []
        self.plant_list = []
        self.nut_list = []
        self.fruit_list = []


        ###########################################################################################################
        # Drinks
        ###########################################################################################################

        # possible drink categories
        self.drink_category = ['water','juice','milk']

        self.food_list = []
        self.food_stored = 0
        self.consumption = []
        self.hunting_method_list = []

        # possible locations in the world
        self.location_list = ['river', 'tent', 'fire']

        # an epoch is a period staring from the initialization of the event tree of a human, until the event tree get
        # completed, this is well defined since currently only one human is generated in the world
        self.epoch = 0


        # total behavior count for all creatures
        self.eat_count_meal = 0
        self.eat_count_fruit = 0
        self.eat_count_nut = 0
        self.eat_count_plant = 0
        self.drink_count = 0
        self.sleep_count = 0
        self.idle_count = 0

        # amount of behavior by category
        self.carnivore_eat = 0
        self.herbivore_eat = 0
        self.human_eat = 0

        ################################################################################################################
        # entity taxonomy
        ################################################################################################################
        self.noun_tax = {'Jessica':'human','Jon':'human','Anastasia':'human','Phil':'human','Andrew':'human',
                         'Lin Khern':'human','Emily':'human','Shufan':'human','Jacki':'human','Katherine':'human',
                        'rabbit':'herb_s','squirrel':'herb_s','fox':'herb_s',
                         'mouflon':'herb_m', 'ibex':'herb_m', 'boar':'herb_m',
                         'bison':'herb_l','buffalo':'herb_l','auroch':'herb_l',
                         'wolf':'carnivore','tiger':'carnivore','hyena':'carnivore',
                         'apple':'fruit','peach':'fruit','pear':'fruit',
                         'walnut':'nut', 'cashew':'nut', 'almond':'nut',
                         'leaf':'plant','grass':'plant','flower':'plant',
                         'water':'drink', 'juice':'drink', 'milk':'drink',
                         'river':'location', 'tent':'location','fire':'location'}

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
        self.pairs = {} # all pairs
        self.noun_stems = [] # noun wthout thematic roles
        self.noun_dict = {} # dictionary for noun stems and their roles


    def create_humans(self):  # humans are generated once the world is there, the current setting is a hunter-gatherer's
        # world
        name_list = random.sample(self.name_list, config.World.num_humans)
        for name in name_list:
            a_human = Human(self, name)
            if config.World.event_tree_human == 'src/event_tree_human.txt':
                a_human.get_hunting_method()
            self.human_list.append(a_human)
            self.agent_list.append(a_human)


    def create_herbivores(self):
        for herbivore in self.herbivore_category[:6]:
            for i in range(config.World.num_herbivores):
                a_herbivore = animal.Herbivore(self,herbivore)
                self.herbivore_list.append(a_herbivore)
                self.agent_list.append(a_herbivore)

    def create_carnivores(self):
        for carnivore in self.carnivore_category[:2]:
            for i in range(config.World.num_carnivores):
                a_carnivore = animal.Carnivore(self,carnivore)
                self.carnivore_list.append(a_carnivore)
                self.agent_list.append(a_carnivore)

    def create_plant(self):
        for i in range(config.World.num_plants)[:2]:
            a_plant = plant.Plant(self)
            self.plant_list.append(a_plant)
            self.plant_resource.append(a_plant)

    def create_nut(self):
        for i in range(config.World.num_nuts)[:2]:
            a_nut = plant.Nut(self)
            self.nut_list.append(a_nut)
            self.plant_resource.append(a_nut)

    def create_fruit(self):
        for i in range(config.World.num_fruits)[:2]:
            a_fruit = plant.Fruit(self)
            self.fruit_list.append(a_fruit)
            self.plant_resource.append(a_fruit)


    def next_turn(self): # human and animals take turns in order
        for human in self.human_list:
            human.take_turn()

        for carnivore in self.carnivore_list:
            carnivore.take_turn()

        for herbivore in self.herbivore_list:
            herbivore.take_turn()

        for plant_r in self.plant_resource:
            plant_r.grow()




        # hunt
        #   location: where animal is
        #   instruments: bows, arrows, spears, hand axe, trap

        # cooking
        #   # location: hearth
        #   instruments: fire, spear

        # sleep
        #   location: hut
        #   instruments: blanket


