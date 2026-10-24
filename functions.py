import numpy as np
import torch

from Games.Go9x9.environment import State as Go9x9_State, Game as Go9x9_Game
from Games.Go9x9.representation import state_to_tensor as go9x9_state_to_tensor

class State():
    def __new__(self, environment):
        if environment == "Go9x9":
            return Go9x9_State
 
class Game():
    def __new__(self, environment):
        if environment == "Go9x9":
            return Go9x9_Game