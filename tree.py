import torch
 
import numpy as np
import math
import json
import copy
import importlib

def PUCT(parent, child, exploration_constant): # parent and child are "Node" objects
    # +--- Compute Prior/Exploration (Right of Equation) ----+
    prior_score = child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    prior_score *= exploration_constant
 
    # +--- Compute Value (Left of Equation) ----+
    value_score = child.compute_value()
 
    return value_score + prior_score

class Node: # initialise on discovery
    def __init__(self, prior, to_move, search):
        self.prior = prior
        self.to_move = to_move
        self.search = search
        self.game = None
 
        self.value_sum = 0.
        self.visit_count = 0
        self.parent = None
        self.parent_action = None
        self.children = []
 
        self.is_terminal = False
        self.terminal_value = 0.
        self.is_expanded = False

    def expand(self, game, output):
        self.is_expanded = True
        self.game = game
        self.to_move = game.state.to_move
 
        nn_dist = output[:-1]
        nn_value = output[-1].item()
 
        self.children = [None] * len(nn_dist)
        for child_index, probability in enumerate(nn_dist.tolist()):
            if probability > 0:
                child = Node(prior=probability, to_move=None, search=self.search)
                child.parent = self
                child.parent_action = child_index
                self.children[child_index] = child
 
        return nn_value, nn_dist

    def compute_value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count
 
    def compute_best_child(self):
        c = self.search.exploration_constant
        return max((child for child in self.children if child is not None),
                   key=lambda child: PUCT(self, child, c))

class TreeSearch: # changed to be GPU-friendly
    def __init__(self, environment):
        self.environment = environment
 
        representation = importlib.import_module(f"Games.{environment}.representation")
        self.state_to_tensor = representation.state_to_tensor
 
        with open(f"Games/{environment}/config.json") as file:
            config = json.load(file)
 
        self.hyperparameters = config["hyperparameters"]
        self.alpha = self.hyperparameters["alpha"]
        self.epsilon = self.hyperparameters["epsilon"]
        self.exploration_constant = self.hyperparameters["exploration_constant"]
        self.search_depth = self.hyperparameters["search_depth"]
 
    def start(self, game): # initialises new tree search from a root. root to be evaluated by driver ..
        self.root = Node(prior=None, to_move=None, search=self)
        self.simulations = 0
        self.leaf = self.root
        self.leaf_game = copy.deepcopy(game)
        self.path = [self.root]
        return self.leaf_game.state

    def next_leaf(self): # simply runs down the tree until a "new" node (that model hasn't performed inference on) is found
        while self.simulations < self.search_depth:
            current_node = self.root
            path = [current_node]

            while current_node.is_expanded:
                current_node = current_node.compute_best_child()
                path.append(current_node)

            if current_node.is_terminal:
                self.backup(path, current_node, current_node.terminal_value)
                continue

            child_game = copy.deepcopy(current_node.parent.game)
            child_game.play(child_game.index_to_move(current_node.parent_action))

            if child_game.is_over():
                current_node.is_terminal = True
                current_node.game = child_game
                current_node.to_move = child_game.state.to_move

                if child_game.result == 0:
                    current_node.terminal_value = 0.
                elif child_game.result == current_node.to_move:
                    current_node.terminal_value = 1.
                else:
                    current_node.terminal_value = -1.

                self.backup(path, current_node, current_node.terminal_value)
                continue

            self.leaf, self.leaf_game, self.path = current_node, child_game, path
            return child_game.state

        return None

    def receive(self, output):
        value, _ = self.leaf.expand(self.leaf_game, output)

        if self.leaf is self.root:
            self.root_output = output
            legal = [child for child in self.root.children if child is not None]
            nablas = np.random.dirichlet([self.alpha] * len(legal))
            for child, nabla in zip(legal, nablas):
                child.prior = (1 - self.epsilon) * child.prior + self.epsilon * nabla
        else:
            self.backup(self.path, self.leaf, value)

    def backup(self, path, leaf, value):
        for history_node in path:
            sign = -1 if history_node.to_move == leaf.to_move else 1
            history_node.visit_count += 1
            history_node.value_sum += sign * value
        self.simulations += 1

    def result(self):
        visits = np.array([child.visit_count if child is not None else 0
                           for child in self.root.children], dtype=np.float32)
        policy = visits / visits.sum()
        return np.append(policy, -self.root.compute_value()).astype(np.float32)