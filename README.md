# MIMO-pytorch

PyTorch implementation of MIMO proposed in [Training independent subnetworks for robust prediction](https://openreview.net/forum?id=OGg9XnKxFAH).

# Usage

## Baseline Model Training

``` sh
python baseline_train.py
```

## MIMO Model Training

``` sh
python mimo_train.py
```

# References

``` plain
@inproceedings{havasi2021training,
  author = {Marton Havasi and Rodolphe Jenatton and Stanislav Fort and Jeremiah Zhe Liu and Jasper Snoek and Balaji Lakshminarayanan and Andrew M. Dai and Dustin Tran},
  title = {Training independent subnetworks for robust prediction},
  booktitle = {International Conference on Learning Representations},
  year = {2021},
}
```

* https://github.com/google/edward2/tree/master/experimental/mimo
