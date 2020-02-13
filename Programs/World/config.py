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
    num_humans = 1
    num_animals = 100
    num_turn = 10000
    event_tree_file = 'World/event_tree.txt'
