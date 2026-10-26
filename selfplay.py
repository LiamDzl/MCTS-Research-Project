import time
import torch
import json
import sys
import os
import importlib

from selfplay_functions import run_selfplay

device = "cuda" if torch.cuda.is_available() else "cpu"

# --- SET ++
environments = ["Connect4", "Breakthrough", "Go9x9"]
env = environments[1]

RUN_NUM = 0

# ------+

worker_id = sys.argv[1]

data_dir = f"Games/{env}/Runs/Run{RUN_NUM}/Datasets"
os.makedirs(data_dir, exist_ok=True)

representation = importlib.import_module(f"Games.{env}.representation")
state_to_tensor = representation.state_to_tensor

with open(f"Games/{env}/config.json") as file:
    config = json.load(file)

model = torch.load(f"Games/{env}/Runs/Run{RUN_NUM}/Models/gen_0.pt",
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

with open(f"{data_dir}/W_{worker_id}.json", "w") as file:
    json.dump(dataset, file)

end = time.time()

print(end - start)