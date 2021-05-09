import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMOModel(nn.Module):
    def __init__(self, ensemble_num: int = 3):
        super(MIMOModel, self).__init__()
        self.cnn_layer = CNNLayer()
        self.ensemble_num = ensemble_num
        self.last_head = nn.Linear(128, 10 * ensemble_num)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        input_shape_list = list(input_tensor.size())  # (ensemble_num,batch_size,1,28,28)
        ensemble_num, batch_size = input_shape_list[0], input_shape_list[1]
        assert ensemble_num == self.ensemble_num

        input_tensor = input_tensor.view([ensemble_num * batch_size] + input_shape_list[2:])
        output = self.cnn_layer(input_tensor)
        output = output.view(ensemble_num, batch_size, -1)
        output = self.last_head(output)
        output = output.view(ensemble_num, batch_size, ensemble_num, -1)
        output = torch.diagonal(output, offset=0, dim1=0, dim2=2).permute(2, 0, 1)
        output = F.log_softmax(output, dim=-1)
        return output


# from https://github.com/pytorch/examples/blob/master/mnist/main.py
class CNNLayer(nn.Module):
    def __init__(self):
        super(CNNLayer, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
