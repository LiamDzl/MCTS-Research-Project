import time
import torch
import json
import sys

from selfplay_functions import run_selfplay

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SET ++
environments = ["Connect4", "Breakthrough", "Go9x9"]
env = environments[1]

RUN_NUM = 0

# ------+

if env == "Breakthrough":
    from Games.Breakthrough.representation import state_to_tensor

with open(f"Games/{env}/config.json") as file:
            config = json.load(file)

model = torch.load(f"Games/{env}/Runs/Run0/Models/gen_0.pt",
                   weights_only=False, map_location=device)

model.eval()

start = time.time()

dataset = run_selfplay(model=model,
                    env=env,
                    temp_for=config["hyperparameters"]["temp_for"],
                    state_to_tensor=state_to_tensor,
                    device=device,
                    n_parallel=16,
                    n_total=100)

worker_id = sys.argv[1]

with open(f"Games/{env}/Runs/Run{RUN_NUM}/Data/worker_{worker_id}.json", "w") as file:
    json.dump(dataset, file)

end = time.time()

print(end-start)