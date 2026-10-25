import time
import torch
import json

from selfplay_functions import run_selfplay

device = "cuda" if torch.cuda.is_available() else "cpu"

environments = ["Connect4", "Breakthrough", "Go9x9"]
env = environments[1]

if env == "Breakthrough":
    from Games.Breakthrough.representation import state_to_tensor, to_display

with open(f"Games/{env}/config.json") as file:
            config = json.load(file)

model = torch.load(f"Games/{env}/Runs/Run0/Models/gen_0.pt",
                   weights_only=False, map_location=device)

for var in [16, 32, 64, 128, 256, 512]:

    start = time.time()

    dataset = run_selfplay(model=model,
                        env=env,
                        temp_for=config["hyperparameters"]["temp_for"],
                        state_to_tensor=state_to_tensor,
                        device=device,
                        n_parallel=var,
                        n_total=var)

    end = time.time()

    print(var)
    print(end-start)