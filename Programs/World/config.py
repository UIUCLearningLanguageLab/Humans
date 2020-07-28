class World:

    ####################################################################################################################
    # Basic parameters for the world
    # In the current setting, there is only one human in the world, but there are multiple animals
    # num_turn stipulates the number of turns that the agents in the world takes in each trial.
    # event_tree_file specifies which event tree the human obeys.
    ####################################################################################################################

    tile_size = 50
    num_tiles = 50
    world_size = tile_size * num_tiles
    num_humans = 2
    num_herbivores = 10
    num_carnivores = 1
    num_nuts = 100
    num_fruits = 100
    num_plants = 100
    num_turn = 10000
    event_tree_human = 'World/event_tree_human.txt'
    event_tree_carnivore = 'World/event_tree_carnivore.txt'
    event_tree_herbivore = 'World/event_tree_herbivore.txt'
