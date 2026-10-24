import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np
import json

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

class Resnet(nn.Module):
    def __init__(self, environment):
        super().__init__()

        with open(f"Games/{environment}/config.json") as file:
            config = json.load(file)
            self.architecture = config["architecture"]

        # Upscale:
        self.conv_up = nn.Conv2d(in_channels=self.architecture["upscale"][0],
                                 out_channels=self.architecture["upscale"][1],
                                 kernel_size=3,
                                 padding=1, bias=False)

        # ResBlocks: 
        self.convs = []
        self.bns = []

        for _ in range(2 * self.architecture["residual_blocks"]): # * 2 since each res. block contains two conv. mappings
            conv = nn.Conv2d(in_channels=self.architecture["residual_dimension"],
                             out_channels=self.architecture["residual_dimension"],
                             kernel_size=3,
                             padding=1, bias=False)

            bn = nn.BatchNorm2d(self.architecture["residual_dimension"])

            self.convs.append(conv)
            self.bns.append(bn)

        # Downscale:
        self.conv_down = nn.Conv2d(in_channels=self.architecture["downscale"][0],
                                 out_channels=self.architecture["downscale"][1],
                                 kernel_size=3,
                                 padding=1, bias=False)
        
        self.bn_down = nn.BatchNorm2d(self.architecture["downscale"][1])

        # Policy and Value Heads:
        self.policy_maps = []
        self.value_maps = []

        for index in range(len(self.architecture["policy_architecture"])-1):
            pm = nn.Linear(self.architecture["policy_architecture"][index],
                           self.architecture["policy_architecture"][index+1])

            vm = nn.Linear(self.architecture["value_architecture"][index],
                           self.architecture["value_architecture"][index+1])

            self.policy_maps.append(pm)
            self.value_maps.append(vm)

        self.convs = nn.ModuleList(self.convs)
        self.bns = nn.ModuleList(self.bns)
        self.policy_maps = nn.ModuleList(self.policy_maps)
        self.value_maps = nn.ModuleList(self.value_maps)

    def forward(self, x, mask):
        # Forward Conv, Upscale.
        x = self.conv_up(x)

        # ResBlocks:
        for index in range(self.architecture["residual_blocks"]):
            res = x
            x = self.convs[2*index](F.relu(self.bns[2*index](x)))
            x = self.convs[2*index + 1](F.relu(self.bns[2*index + 1](x)))
            x = x + res

        # Downscale
        x = F.relu(self.bn_down(self.conv_down(x)))
        x = x.reshape(x.shape[0], -1)

        # Policy Head
        policy = x

        for index in range(len(self.policy_maps)-1):
            policy = self.policy_maps[index](policy)
            policy = F.relu(policy)

        policy = self.policy_maps[-1](policy)

        # Policy Mask
        mask = torch.as_tensor(mask, dtype=policy.dtype, device=policy.device).reshape(policy.shape)
        policy = policy + mask
        policy = F.softmax(policy, dim=1)

        # Value Head
        value = x
        
        for index in range(len(self.value_maps)-1):
            value = self.value_maps[index](value)
            value = F.relu(value)

        value = self.value_maps[-1](value)
        value = F.tanh(value)

        output = torch.cat([policy, value.unsqueeze(0)[0]], dim=1)

        return output