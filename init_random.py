import torch
from neural_network import Resnet

environments = ["Connect4", "Breakthrough", "Go9x9"]
env = environments[1]

resnet = Resnet(environment=env)

torch.save(resnet, f"Games/{env}/Runs/Run0/Models/gen_0.pt")