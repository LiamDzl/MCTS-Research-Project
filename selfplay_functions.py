import json
import importlib
import numpy as np
import torch

from functions import Game, outcome_fn
from tree import TreeSearch

class SelfPlayGame:
    def __init__(self, env, temp_for, display=False, to_display=None):
        self.rng = np.random.default_rng()
        self.temp_for = temp_for
        self.display = display
        self.to_display = to_display

        self.game = Game(environment=env)()
        self.tree = TreeSearch(environment=env)
        self.tree_distrs = []

        self.waiting_on = self.tree.start(self.game)

    def give(self, output):
        self.tree.receive(output)
        self.waiting_on = self.tree.next_leaf()

        if self.waiting_on is None:
            self.play_move()
            if not self.game.is_over():
                self.waiting_on = self.tree.start(self.game)

    def play_move(self):
        if self.display:
            self.game.state.show(prior=self.to_display(self.tree.root_output),
                                 posterior=self.to_display(self.tree.result()))

        tree_distr = self.tree.result()[:-1]
        self.tree_distrs.append(tree_distr)

        if len(self.game.moves) < self.temp_for:
            index = self.rng.choice(len(tree_distr), p=tree_distr)
        else:
            index = np.argmax(tree_distr)

        self.game.play(self.game.index_to_move(index))

    def dataset(self):
        dataset = []
        for index, state in enumerate(self.game.trajectory[:-1]):
            data = {
                    "state": state,
                    "search": self.tree_distrs[index],
                    "outcome": outcome_fn(to_move=state.to_move,
                                          outcome=self.game.result),
                    }
            dataset.append(data)
        return dataset

def evaluate(model, states, state_to_tensor, device):
    Xs = torch.cat([state_to_tensor(s) for s in states]).to(device)
    masks = torch.cat([torch.as_tensor(s.mask).reshape(1, -1) for s in states]).to(device)

    with torch.inference_mode():
        out = model(Xs, masks)

    return out.cpu()

def play_single(model, env, temp_for, state_to_tensor, device, display=False, to_display=None):
    selfplay = SelfPlayGame(env, temp_for, display=display, to_display=to_display)

    while not selfplay.game.is_over():
        out = evaluate(model, [selfplay.waiting_on], state_to_tensor, device)
        selfplay.give(out[0])

    return selfplay.dataset()

def run_selfplay(model, env, temp_for, state_to_tensor, device, n_parallel, n_total):
    games = [SelfPlayGame(env, temp_for) for _ in range(min(n_parallel, n_total))]
    games_started = len(games)
    dataset = []

    while games:
        out = evaluate(model, [g.waiting_on for g in games], state_to_tensor, device)
        for g, o in zip(games, out):
            g.give(o)

        still_running = []
        for g in games:
            if g.game.is_over():
                dataset.extend(g.dataset())
                if games_started < n_total:
                    still_running.append(SelfPlayGame(env, temp_for))
                    games_started += 1
            else:
                still_running.append(g)
        games = still_running

    return dataset