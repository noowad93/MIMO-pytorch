from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from mimo.config import Config


class MIMOTrainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_dataloaders: List[DataLoader],
        test_dataloader: DataLoader,
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.train_dataloaders: List[DataLoader] = train_dataloaders
        self.test_dataloader: DataLoader = test_dataloader

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=len(self.train_dataloaders[0]), gamma=config.gamma)

        self.device = device

    def train(self):
        self.model.to(self.device)
        self.model.train()
        global_step = 0
        for epoch in range(1, self.config.num_epochs + 1):
            for datum in zip(*self.train_dataloaders):
                model_inputs = torch.stack([data[0] for data in datum]).to(self.device)
                targets = torch.stack([data[1] for data in datum]).to(self.device)

                ensemble_num, batch_size = list(targets.size())
                self.optimizer.zero_grad()
                outputs = self.model(model_inputs)
                loss = F.nll_loss(
                    outputs.reshape(ensemble_num * batch_size, -1), targets.reshape(ensemble_num * batch_size)
                )
                loss.backward()

                self.optimizer.step()
                self.scheduler.step()

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
            for data in self.test_dataloader:
                model_inputs = torch.stack([data[0]] * self.config.ensemble_num).to(self.device)
                target = data[1].to(self.device)

                outputs = self.model(model_inputs)
                output = torch.mean(outputs, axis=1)

                test_loss += F.nll_loss(output, target, reduction="sum").item()
                pred = output.argmax(dim=-1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloader.dataset)
        acc = 100.0 * correct / len(self.test_dataloader.dataset)
        print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")
        self.model.train()
