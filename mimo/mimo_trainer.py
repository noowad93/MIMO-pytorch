from typing import Union, List
import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
from mimo.config import Config
from torch.optim.lr_scheduler import StepLR


class MIMOTrainer:
    def __init__(
        self,
        config: Config,
        model: nn.Module,
        train_dataloaders: List[DataLoader],
        test_dataloaders: List[DataLoader],
        device: torch.device,
    ):
        self.config = config
        self.model = model
        self.train_dataloaders = train_dataloaders
        self.test_dataloaders = test_dataloaders

        self.optimizer = optim.Adadelta(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=config.gamma)

        self.device = device


    def train(self):
        self.model.to(self.device)
        self.model.train()
        global_step = 0
        for epoch in range(1, self.config.num_epochs + 1):
            for datum in zip(*self.train_dataloaders):
                model_inputs = [data[0].to(self.device) for data in datum]
                targets = [data[1].to(self.device) for data in datum]

                self.optimizer.zero_grad()
                outputs = self.model(model_inputs)
                losses = [F.nll_loss(output, target) for output,target in zip(outputs,targets)]
                [loss.backward() for loss in losses]
                self.optimizer.step()
                loss=torch.mean(torch.stack(losses))

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
            for datum in zip(*self.test_dataloaders):
                model_inputs = [data[0].to(self.device) for data in datum]
                target = datum[0][1].to(self.device)

                outputs = self.model(model_inputs)
                output = torch.mean(torch.stack(outputs),axis=0)
                test_loss += F.nll_loss(output, target, reduction="sum").item()  # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(self.test_dataloaders[0].dataset)
        acc = 100.0 * correct / len(self.test_dataloaders[0].dataset)
        print(f"[Valid] Average loss: {test_loss:.4f} \t Accuracy:{acc:2.2f}%")
        self.model.train()
