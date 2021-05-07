from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader
from mimo.config import Config
from mimo.model import MIMOModel
from mimo.mimo_trainer import MIMOTrainer


def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST("../data", train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST("../data", train=False, transform=transform)

    train_dataloaders = [
        DataLoader(
            train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True, shuffle=True
        )
        for _ in range(config.ensemble_num)
    ]
    test_dataloaders = [
        DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, pin_memory=True)
    ] * config.ensemble_num

    model = MIMOModel(config.ensemble_num).to(device)
    trainer = MIMOTrainer(config, model, train_dataloaders, test_dataloaders, device)
    trainer.train()


if __name__ == "__main__":
    main()
