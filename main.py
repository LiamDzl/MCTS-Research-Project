import torch
import torch.nn.functional as F
import numpy as np
import sys

from functions import Game, parse
from neural_network import Resnet
from tree import TreeSearch

env = input("\n🌳 Select Environment: ")
env, player = parse(env)

resnet = Resnet(environment=env)
game = Game(environment=env)()
tree = TreeSearch(environment=env, model=resnet)

if env == "Go9x9":
    from Games.Go9x9.representation import state_to_tensor, to_display
    output_size = 83
elif env == "Breakthrough":
    from Games.Breakthrough.representation import state_to_tensor, to_display
    output_size = 65

while game.is_over() == False:
    if player == 1 and game.moves == []:
        game.state.show(prior=torch.zeros(output_size), posterior=torch.zeros(output_size))

        boo = False

        print("")
        while boo is False:    
            move = game.get_human_move()
                    
            if move == "end":
                print("")
                sys.exit()

            boo = game.play(move)

        print("\n... Thinking ...")

    elif player == 2 and game.moves == []:
        print("\n... Thinking ...")

    x = state_to_tensor(game.state)
    mask = game.state.mask

    nn_output = resnet.forward(x, mask)
    tree_output = tree.search(game=game)

    index = np.argmax(tree_output[:-1])
    argmax_move = game.index_to_move(index)
    game.state.show(prior=to_display(nn_output), posterior=to_display(tree_output))
    game.play(argmax_move)

    print("\n+-----+")
    game.state.show(prior=torch.zeros(output_size), posterior=torch.zeros(output_size))

    boo = False

    print("")
    while boo is False:  
        move = game.get_human_move()
                
        if move == "end":
            print("")
            sys.exit()
        
        boo = game.play(move)

    print("\n... Thinking ...")

    