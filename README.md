# pytorch-xla-transformer-language-model
This repository is an open source test case for [pytorch/xla](https://github.com/pytorch/xla) that runs a minimal training loop for a [Transformer](https://arxiv.org/abs/1706.03762) language model on a single TPU device.

This code is intended to be used as reference for testing the compilation of the model by XLA, and is not intended to be used for training a reasonable language model. During initial runs, this code triggered recompilation far too often, but these issues have now been resolved. 

Depends on Docker image `gcr.io/tpu-pytorch/xla:r0.1`.

```bash
export TPU_IP=#YOU MUST SET YOUR TPU IP
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP"
export XLA_USE_32BIT_LONG=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
python3 train.py
```
Output is in `run.log`.
