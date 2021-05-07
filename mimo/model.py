from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F

# from https://github.com/pytorch/examples/blob/master/mnist/main.py
class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.last_head = nn.Linear(128, 10)

    def main_forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

    def forward(self, input_tensor: torch.Tensor):
        output = self.main_forward(input_tensor)
        output = self.last_head(output)
        output = F.log_softmax(output, dim=1)
        return output

class MIMOModel(nn.Module):
    def __init__(self, ensemble_num:int=3):
        super(MIMOModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.last_heads = nn.ModuleList([nn.Linear(128, 10) for _ in range(ensemble_num)])

    def main_forward(self, x:torch.Tensor)->torch.Tensor:
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        return x

    def forward(self, input_tensors: List[torch.Tensor])->List[torch.Tensor]:
        outputs:List[torch.Tensor]=[]
        for idx, input_tensor in enumerate(input_tensors):
            output = self.main_forward(input_tensor)
            output = self.last_heads[idx](output)
            output = F.log_softmax(output, dim=1)
            outputs.append(output)
        return outputs
