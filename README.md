# pytorch-xla-transformer-language-model
This repository is an open source test case for [pytorch/xla](https://github.com/pytorch/xla) that runs a minimal training loop for a [Transformer](https://arxiv.org/abs/1706.03762) language model on a single TPU device.

This tests the compilation of the model by XLA, and is not intended to be used for training a reasonable language model.

Depends on Docker image `gcr.io/tpu-pytorch/xla:r0.1`.

Run with `python3 train.py`.

Output is in `run.log`.
