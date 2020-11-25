# MrAP
This folder contains the implementation of the multi-relational attribute propagation algorithm.

It requires the package [pytorch-scatter](https://pytorch-scatter.readthedocs.io/en/latest/index.html).

Please check out the notebooks for the usage. The experiments are executed on version 1.6.0+cu101 of PyTorch and version 2.0.4 of PyTorch-scatter.

# Data
You can download the Knowledge Base data enriched with numerical node attributes ( FB15K-237, YAGO15K) from https://github.com/nle-ml/mmkb.

# License
MIT

Please cite our paper if you use the code:
```
@article{bayram2020nodeattributecompletion,
    title={Node Attribute Completion in Knowledge Graphs with Multi-Relational Propagation},
    author={Eda Bayram and Alberto Garcia-Duran and Robert West},
    journal={arXiv preprint arXiv:2011.05301},
    year={2020}
}
```
