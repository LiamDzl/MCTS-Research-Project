import json
import importlib
import torch

from functions import parse
from selfplay_functions import play_single

env = input("\n🌳 Select Environment (Connect 4, Breakthrough or 9x9 Go): ")
env, player = parse(env, spectate_mode=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
representation = importlib.import_module(f"Games.{env}.representation")
environment = importlib.import_module(f"Games.{env}.environment")

with open(f"Games/{env}/config.json") as file:
    temp_for = json.load(file)["hyperparameters"]["temp_for"]

model = torch.load(f"Games/{env}/Runs/Run0/Models/gen_0.pt",
                   weights_only=False, map_location=device)

model.eval().to(device)

dataset = play_single(model, env, temp_for, representation.state_to_tensor, device,
                      display=True, to_display=representation.to_display)

for data in dataset:
    environment.show(data["state"].board)
    print(data["search"])
    print(data["outcome"])