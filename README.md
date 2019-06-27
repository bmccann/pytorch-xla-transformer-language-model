```bash
export TPU_IP=#YOU MUST SET YOUR TPU IP
export XRT_TPU_CONFIG="tpu_worker;0;$TPU_IP"
export XLA_USE_32BIT_LONG=1
export XLA_IR_DEBUG=1
export XLA_HLO_DEBUG=1
python train.py
```
