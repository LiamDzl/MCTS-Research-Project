import numpy as np
import torch

from Games.Breakthrough.environment import State as Breakthrough_State, Game as Breakthrough_Game
from Games.Go9x9.environment import State as Go9x9_State, Game as Go9x9_Game

environments = ["Connect4", "Breakthrough", "Go9x9"]

class State():
    def __new__(self, environment):
        if environment == "Go9x9":
            return Go9x9_State
        elif environment == "Breakthrough":
            return Breakthrough_State
 
class Game():
    def __new__(self, environment):
        if environment == "Go9x9":
            return Go9x9_Game
        elif environment == "Breakthrough":
            return Breakthrough_Game

def parse(user_text, spectate_mode=False):
    if user_text in {"go", "Go", "9x9Go", "Go9x9", "9x9go", "go9x9", "9x9 Go", "9x9 go", "go 9x9", "Go 9x9", "9x9"}:
        env = environments[2] # 9x9 Go

        if spectate_mode == False:
            colour = input("\n⚫ / ⚪ ? : ")
            if colour in {"w", "white", "White"}:
                player = 2
            else:
                player = 1
        else:
            player = None

        return env, player

    elif user_text in {"break", "breakthrough", "br", "b", "Breakthrough", "Break"}:
        env = environments[1] # Breakthrough
        print("")
        print(r""" ____                 _    _   _                           _       
| __ ) _ __ ___  __ _| | _| |_| |__  _ __ ___  _   _  __ _| |__    
|  _ \| '__/ _ \/ _` | |/ / __| '_ \| '__/ _ \| | | |/ _` | '_ \   
| |_) | | |  __/ (_| |   <| |_| | | | | | (_) | |_| | (_| | | | |_ 
|____/|_|  \___|\__,_|_|\_\\__|_| |_|_|  \___/ \__,_|\__, |_| |_(_)
                                                     |___/         """)

        if spectate_mode == False:
            colour = input("\n⚫ / ⚪ ? : ")
            if colour in {"b", "black", "Black"}:
                player = 2
            else:
                player = 1
        else:
            player = None

        return env, player

    else:
        return parse(input("\n🤔 Hmm.. Try Again? "), spectate_mode)

def outcome_fn(to_move, outcome):
    if outcome == 0:
        return 0
    elif to_move == outcome:
        return 1
    else:
        return -1