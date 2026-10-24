import torch
import torch.nn.functional as F
import numpy as np
import sys

from functions import Game
from neural_network import Resnet
from tree import TreeSearch

environments = ["Connect4", "Breakthrough", "Go9x9"]

env = input("\n🌳 Select Environment: ")

if env in {"go", "Go", "9x9Go"}:
    environment = environments[2] # 9x9 Go
    colour = input("\n⚫ / ⚪ ? : ")
    if colour in {"b", "black", "Black"}:
        player = 1
    else:
        player = 2

resnet = Resnet(environment=environment)
game = Game(environment=environment)()
tree = TreeSearch(environment=environment, model=resnet)

if environment == "Go9x9":
    from Games.Go9x9.representation import state_to_tensor

while game.is_over() == False:
    if player == 1 and game.moves == []:
        game.state.show(prior=torch.zeros(83), posterior=torch.zeros(83))

        boo = False

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

    game.state.show(prior=nn_output, posterior=tree_output)
    game.play(argmax_move)

    print("+-----+")
    game.state.show(prior=torch.zeros(83), posterior=torch.zeros(83))

    boo = False
    
    while boo is False:    
        move = game.get_human_move()
                
        if move == "end":
            print("")
            sys.exit()
        
        boo = game.play(move)

    print("\n... Thinking ...")

    