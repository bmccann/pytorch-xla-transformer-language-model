import os
import time
import math

import numpy
import torch
import torch.utils.data

import torch_xla
import torch_xla_py.xla_model as xm
import torch_xla_py.data_parallel as dp

from transformer import Transformer


class LazyDataset:

  def __init__(self, path, sequence_length):
    self.path = path
    self.size = os.stat(path).st_size - sequence_length
    self.sequence_length = sequence_length 

  def __getitem__(self, index):
    with open(self.path, 'rb') as f:
      f.seek(index)
      chunk = f.read(self.sequence_length)
    return torch.ByteTensor(numpy.frombuffer(chunk, dtype=numpy.uint8))

  def __len__(self):
    return self.size


def get_dataloader(path, batch_size, sequence_length, num_workers):
  dataset = LazyDataset(path, sequence_length+1)
  sampler = torch.utils.data.RandomSampler(dataset)
  return torch.utils.data.DataLoader(dataset, sampler=sampler, batch_size=batch_size, num_workers=num_workers)


def main():
  device = xm.xla_device()
  torch_xla._XLAC._xla_set_default_device(str(device))
  model = Transformer(256, 12, 512, 2048, 8, 0.2)
  model = model.to(device)
  start_time = time.time()
  print('Model on device ', model.embed.weight.device, round(time.time() - start_time, 4))
  optimizer = torch.optim.SGD(model.parameters(), lr=0.001)
  loader = get_dataloader('datasets/enwik8/train/train.txt.raw', 128, 256, 3)
  for iteration, batch in enumerate(loader):
    print('Iteration: ', iteration, round(time.time() - start_time, 4))
    if iteration > 999: return
    batch = batch[1] if isinstance(batch, tuple) else batch
    optimizer.zero_grad()
    print('Finished zeroing grad', round(time.time() - start_time, 4))
    input = batch[:, :-1].long().to(device)
    target = batch[:, 1:].long().to(device)
    batch_size, sequence_length = input.size()
    positions = torch.arange(input.size(1)).long().view(1, sequence_length).to(device)
    causal_mask = torch.triu(torch.ones(sequence_length, sequence_length, dtype=torch.uint8), 
            diagonal=1).unsqueeze(0).to(device)
    print('Finished moving inputs to device ', device, round(time.time() - start_time, 4))
    loss = model(input, positions, target, batch_mask=causal_mask)
    print('Finished forward pass', round(time.time() - start_time, 4))
    loss.backward()
    print('Finished backward pass', round(time.time() - start_time, 4))
    xm.optimizer_step(optimizer)
    print('Finished optimizer step', round(time.time() - start_time, 4))
    print('Loss', loss.item()/math.log(2), round(time.time() - start_time, 4))
    print(torch_xla._XLAC._xla_metrics_report())
  print('Finished 1000 iterations', round(time.time() - start_time, 4))


if __name__ == '__main__':
  main()
