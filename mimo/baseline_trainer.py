from typing import Union, List
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from mimo.config import Config
from torch.optim.lr_scheduler import StepLR


class BaselineTrainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=config.gamma)

        self.device = device

    def train(self):
        self.model.to(self.device)
        self.model.train()
        global_step = 0
        for epoch in range(1, self.config.num_epochs + 1):
            for data, target in self.train_dataloader:
                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = F.nll_loss(output, target)
                loss.backward()
                self.optimizer.step()

                global_step += 1
                if global_step != 0 and global_step % self.config.train_log_interval == 0:
                    print(f"[Train] epoch:{epoch} \t global step:{global_step} \t loss:{loss:.4f}")
                if global_step != 0 and global_step % self.config.valid_log_interval == 0:
                    self.validate()

    def validate(self):
        self.model.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for data, target in self.test_dataloader:
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        acc = 100.0 * correct / len(self.test_dataloader.dataset)
        print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")
        self.model.train()
