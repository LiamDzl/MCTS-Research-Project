import torch
import torch.nn.functional as F

import numpy as np
import math

from neural_network import state_to_tensor

dirichlet = torch.distributions.Dirichlet(torch.full((7,), 0.13))
epsilon = 0.25
exploration_constant = 1.25
search_depth = 200

def PUCT(parent, child): # parent and child are "Node" objects
    # +--- Compute Prior/Exploration (Right of Equation) ----+
    prior_score = child.prior * math.sqrt(parent.visit_count + 1) / (child.visit_count + 1)
    prior_score *= exploration_constant

    # +--- Compute Value (Left of Equation) ----+
    if child.visit_count > 0:
        value_score = child.value()
    else:
        value_score = 0

    return value_score + prior_score

class Node: # initialise on discovery
    def __init__(self, prior, to_move, model):
        self.prior = prior # parent's probability viewpoint of choosing child (scalar)
        self.to_move = to_move
        self.model = model
        self.state = None

        self.nn_dist = None
        self.value_sum = 0.
        self.visit_count = 0
        self.parent = None # stores single parent ("Node" object)
        self.parent_action = None
        self.children = [None] * 82 # for each possible action

        self.is_terminal = False
        self.is_root = False
        self.is_expanded = False

    def expand(self, state):
        self.is_expanded = True
        self.state = state
        x = state_to_tensor(state)
        output = self.model.forward(x)

        nn_dist = output[:-1]
        nn_value = output[-1]

        for child_index, probability in enumerate(nn_dist):
            if probability != 0:
                self.children[child_index] = Node(prior=probability,
                                                  to_move=2 if self.to_move == 1 else 1,
                                                  model=self.model)
                
                self.children[child_index].parent = self
                self.children[child_index].parent_action = child_index

        return nn_value, nn_dist

    def compute_value(self):
        if self.visit_count == 0:
                return 0
        return self.value_sum / self.visit_count

    def compute_best_child(self):
        selected = None
        for child in self.children:
            if child is not None:
                selected = child
                break
        
        for child in self.children: # compare childrens' PUCT scores
            if child == None:
                pass
            # +--- Maximise PUCT ---+
            elif PUCT(self, child, exploration_constant=exploration_constant) > PUCT(self, selected, exploration_constant=exploration_constant):
                selected = child

        return selected

    def reset_root(self, state):
        self.state = state.detach().clone()

class TreeSearch:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.explored_nodes = set()

    def search(self, game): # keep true game separate to simulated game to compute correct ko 
        # +--- Create Root ---+
        state = game.trajectory[-1] # last position
        root = Node(prior=None, to_move=state.to_move, model=self.model)
        root.is_root = True
        root.expand(state=state)
        self.explored_nodes.add(root)

        # +--- Add Dirichlet Noise to Root ---+
        nablas = dirichlet.sample()
        for index, child in enumerate(root.children):
            try:
                child.prior = (1 - epsilon) * child.prior + epsilon * nablas[index].item()
            except:
                pass

        # +--- Main Search Loop ---+
        for iteration in range(search_depth):
            simulation = game
            to_move = state.to_move
            root.reset_root(state)
            proceed = True
            current_node = root
            path = [current_node]

            # Continuously Check Current Node, Traverse Branch.
            while current_node.is_expanded == True:
                current_node = current_node.best_child()
                path.append(current_node)
                depth += 1

                if current_node.is_terminal == True: # don't expand terminal states
                    proceed = False
                    break

                to_move = 2 if self.to_move == 1 else 1

            # +---  Select New Node and Expand ---+
            if proceed == True:
                parent_node = current_node.parent
                grid = Grid(state=parent_node.state)
                
                state_save = copy.deepcopy(parent_node.state) # debugging error
                grid.action(column=current_node.parent_action) # new state for new node
                parent_node.state = state_save

                nn_value, nn_dist = current_node.expand(state=grid.state) # Expand Node, Store Value to Backprop. up the Branch
                self.explored_nodes.append(current_node)

                nn_value = nn_value.item()
                nn_value *= -1 # correct perspectives

            # Check if Terminal  
            if winner(current_node.state) != 0:
                #graphic(current_node.state)
                #print("")
                current_node.is_terminal = True
                nn_value = 1 # Overwrite Signal. Fix to +1, Backprop will handle +/- 1
    
                colour = -compute_player(current_node.state)
                if colour == 1: # i.e. a win,
                    red_wins += 1

                else:
                    yellow_wins += 1

                total_wins += 1

            # Backpropagate up the Branch
            for history_node in path:
                history_node.visit_count += 1
                history_node.value_sum += (history_node.player * current_node.player) * nn_value


