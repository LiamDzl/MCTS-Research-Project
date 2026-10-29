import torch
from neural_network import Resnet

# --- SET ---+
environments = ["Connect4", "Breakthrough", "Go9x9"]
env = environments[1] # Breakthrough

RUN_NUM = 0

# ------+

resnet = Resnet(environment=env)

torch.save(resnet, f"Games/{env}/Runs/Run{RUN_NUM}/Models/gen_0.pt")