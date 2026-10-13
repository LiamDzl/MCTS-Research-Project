import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

def state_to_tensor(board, to_move):
    opponent_color = 2 if to_move == 1 else 1
    current_player_stones = (board == to_move).astype(np.float32)
    empty_squares = (board == 0).astype(np.float32)
    opponent_stones = (board == opponent_color).astype(np.float32)

    stacked = np.stack([current_player_stones, empty_squares, opponent_stones])

    x = torch.from_numpy(stacked)
    x = x.reshape(1, 3, 9, 9)
    return x

def scalar_mask(mask):
    mask = torch.as_tensor(mask, dtype=torch.bool)
    return torch.where(mask, torch.tensor(0.0), torch.tensor(float('-inf')))

class Resnet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv. Layers ---+

        # Upscale
        self.conv_1 = nn.Conv2d(in_channels=3, out_channels=96, kernel_size=3, padding=1, bias=True)
        self.bn_1 = nn.BatchNorm2d(96)

        # ResBlock 1
        self.conv_2 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_2 = nn.BatchNorm2d(96)
        self.conv_3 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_3 = nn.BatchNorm2d(96)

        # ResBlock 2
        self.conv_4 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_4 = nn.BatchNorm2d(96)
        self.conv_5 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_5 = nn.BatchNorm2d(96)

        # ResBlock 3
        self.conv_6 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_6 = nn.BatchNorm2d(96)
        self.conv_7 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_7 = nn.BatchNorm2d(96)

        # ResBlock 4
        self.conv_8 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_8 = nn.BatchNorm2d(96)
        self.conv_9 = nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, padding=1, bias=False)
        self.bn_9 = nn.BatchNorm2d(96)

        # Downscale
        self.conv_10 = nn.Conv2d(in_channels=96, out_channels=12, kernel_size=3, padding=1, bias=True)
        self.bn_10 = nn.BatchNorm2d(12)

        # Policy and Value Heads
        self.policy_head = nn.Linear(972, 82)
        self.value_head = nn.Linear(972, 1)

    def forward(self, x, mask): # x of shape (batch, 3, 9, 9). mask comes in as matrix of booleans

        # Forward Conv, Upscale.
        x = F.elu(self.bn_1(self.conv_1(x)))

        # ResBlock 1
        res = x
        x = F.elu(self.bn_2(self.conv_2(x)))
        x = self.bn_3(self.conv_3(x))
        x = x + res
        x = F.elu(x)

        # ResBlock 2
        res = x
        x = F.elu(self.bn_4(self.conv_4(x)))
        x = self.bn_5(self.conv_5(x))
        x = x + res
        x = F.elu(x)

        # ResBlock 3
        res = x
        x = F.elu(self.bn_6(self.conv_6(x)))
        x = self.bn_7(self.conv_7(x))
        x = x + res
        x = F.elu(x)

        # ResBlock 4
        res = x
        x = F.elu(self.bn_8(self.conv_8(x)))
        x = self.bn_9(self.conv_9(x))
        x = x + res
        x = F.elu(x)

        # Downscale
        x = F.elu(self.bn_10(self.conv_10(x)))
        x = x.reshape(x.shape[0], -1)

        # Policy Head and Masking
        policy = self.policy_head(x)
        mask = scalar_mask(mask)
        policy = policy + mask
        policy = F.softmax(policy, dim=1)

        # Value Head
        value = self.value_head(x)
        value = F.tanh(value)

        output = torch.cat([policy, value.unsqueeze(0)[0]], dim=1)

        return output
