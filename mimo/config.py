from typing import NamedTuple


class Config(NamedTuple):
    """
    Hyperparameters
    """

    #: random seed
    seed: int = 42
    # training epochs
    num_epochs: int = 10
    # batch size
    batch_size: int = 64
    #: learning rate
    learning_rate: float = 1.0
    #: learning rate step gamma
    gamma: float = 0.7
    #: num workers
    num_workers: int = 10

    train_log_interval: int = 100
    valid_log_interval: int = 1000

    """
    MIMO Hyperparameters
    """
    ensemble_num: int = 5
