import torch
import numpy as np

def state_to_tensor(state):
    opponent_color = 2 if state.to_move == 1 else 1

    # Our Pieces:
    current_player_pieces = (state.board == state.to_move)

    # Enemy Pieces:
    opponent_pieces = (state.board == opponent_color)

    # Empty Squares:
    empty_squares = (state.board == 0)

    # Colour Flag (Constant):
    if state.to_move == 1:
        colour = np.zeros((8, 8))
    else:
        colour = np.ones((8, 8))

    stacked = np.stack([current_player_pieces,
                        opponent_pieces,
                        empty_squares,
                        colour]).astype(np.float32)

    x = torch.from_numpy(stacked)
    x = x.reshape(1, 4, 8, 8)

    return x

def to_display(out):
    out = torch.as_tensor(out, dtype=torch.float32).reshape(-1)
    return torch.cat([out[:192].reshape(64, 3).sum(-1), out[192:]])