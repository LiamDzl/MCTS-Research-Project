
import torch
import torch.nn.functional as F
import numpy as np

from go import Game

game = Game()

prior = torch.randn(83)
prior[:-1] = F.softmax(prior[:-1], dim=0)

posterior = prior

game.state.show(prior=prior, posterior=posterior)

