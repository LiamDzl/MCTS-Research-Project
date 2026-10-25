import torch
import numpy as np

def state_to_tensor(state):
    opponent_color = 2 if state.to_move == 1 else 1

    # Our Stones:
    current_player_stones = (state.board == state.to_move)

    # Enemy Stones:
    opponent_stones = (state.board == opponent_color)

    # Empty Stones:
    empty_squares = (state.board == 0)

    # Colour Flag (Constant):
    if state.to_move == 1:
        colour = np.zeros((9, 9))
    else:
        colour = np.ones((9, 9))

    # Mask (Including Superko):
    mask = (state.mask[:81] < 0).reshape(9, 9)

    # Pass Flag:
    pass_flag = np.full((9, 9), state.last_passed)

    stacked = np.stack([current_player_stones,
                        opponent_stones,
                        empty_squares,
                        colour,
                        mask,
                        pass_flag]).astype(np.float32)

    x = torch.from_numpy(stacked)
    x = x.reshape(1, 6, 9, 9)

    return x

def to_display(out):
    return out