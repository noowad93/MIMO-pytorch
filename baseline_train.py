from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from mimo.config import Config
from mimo.model import Net
from mimo.trainer import Trainer


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True
    )

    model = Net().to(device)
    trainer = Trainer(config, model, train_dataloader, test_dataloader, device)
    trainer.train()


if __name__ == "__main__":
    main()
