import torch
import torch.nn as nn
import torch.nn.functional as F


class MIMOModel(nn.Module):
    def __init__(self, hidden_dim: int = 784, ensemble_num: int = 3):
        super(MIMOModel, self).__init__()
        self.input_layer = nn.Linear(hidden_dim, hidden_dim * ensemble_num)
        self.backbone_model = BackboneModel(hidden_dim, ensemble_num)
        self.ensemble_num = ensemble_num
        self.output_layer = nn.Linear(128, 10 * ensemble_num)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        ensemble_num, batch_size, *_ = list(input_tensor.size())
        input_tensor = input_tensor.transpose(1, 0).view(
            batch_size, ensemble_num, -1
        )  # (batch_size, ensemble_num, hidden_dim)
        input_tensor = self.input_layer(input_tensor)  # (batch_size, ensemble_num, hidden_dim * ensemble_num)

        # usual model forward
        output = self.backbone_model(input_tensor)  # (batch_size, ensemble_num, 128)
        output = self.output_layer(output)  # (batch_size, ensemble_num, 10 * ensemble_num)
        output = output.reshape(
            batch_size, ensemble_num, -1, ensemble_num
        )  # (batch_size, ensemble_num, 10, ensemble_num)
        output = torch.diagonal(output, offset=0, dim1=1, dim2=3).transpose(2, 1)  # (batch_size, ensemble_num, 10)
        output = F.log_softmax(output, dim=-1)  # (batch_size, ensemble_num, 10)
        return output


class BackboneModel(nn.Module):
    def __init__(self, hidden_dim: int, ensemble_num: int):
        super(BackboneModel, self).__init__()
        self.l1 = nn.Linear(hidden_dim * ensemble_num, 256)
        self.l2 = nn.Linear(256, 128)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.l1(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.1)
        x = self.l2(x)
        x = F.relu(x)
        return x
