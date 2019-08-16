
class Human:

    def __init__(self):

        self.id_number = None
        self.name = None
        self.hunger = None
        self.sleepiness = None
        self.fatigue = None
        self.energy = None
        self.health = None
        self.max_speed = None
        self.age = None
        self.size = None
        self.strength = None
        self.intelligence = None


    def get_hunt_success_probs(self, animal):
        best_action = None
        prob = 0
        stab_prob = 0
        shoot_prob = 0
        throw_prob = 0
        trap_prob = 0
        return best_action, prob


    def take_action(self):
        pass

        # given calculations of expectation about actions satisfying drives, choose an action
        '''
        while tiredness < hunger:
        
            if food is stored:
                go to the food
                if the food is not butchered
                    butcher the food
                cook the food
                eat the food
                
            else:
                while animal not found
                    search for animal
                    
                if "good chance of kill":
                    pick appropriate method
                    follow steps to trap, chase, 
            



'''
# what is the high level goal (sleep, eat)
# if sleep, do the sleep steps
# if eat, is there gathered food available



# within those, what are specific goal (eat