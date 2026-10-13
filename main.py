import torch
import torch.nn.functional as F
import numpy as np

from go import Game, State
from neural_network import Resnet

resnet = Resnet()
game = Game()

while game.is_over() == True:
    pass

