import numpy
import torch_xla
import torch


def gelu(x):
  return x * torch.sigmoid(1.702 * x)


class Feedforward(torch.nn.Module):

  def __init__(self, outer_dimension, inner_dimension):
    super().__init__()
    self.linear_in = torch.nn.Linear(outer_dimension, inner_dimension)
    self.linear_out = torch.nn.Linear(inner_dimension, outer_dimension)

  def forward(self, input_sequence):
    return self.linear_out(gelu(self.linear_in(input_sequence)))


class Attention(torch.nn.Module):

  def __init__(self, dimension, num_heads):
    super().__init__()
    self.projection = torch.nn.Linear(dimension, 3 * dimension, bias=False)
    self.softmax = torch.nn.Softmax(dim=-1)
    self.scale = dimension ** -0.5
    self.num_heads = num_heads
    self.batch_mask = None
    self.sequence_masks = None
    self.out = torch.nn.Linear(dimension, dimension, bias=False)

  def apply_masks(self, x, bs, nh, sl, hd):
    neginf = (-torch.from_numpy(numpy.array(numpy.inf)))
    if x.dtype == torch.float16:
      neginf = neginf.half() 
    if self.batch_mask is not None:
      return x.masked_fill(self.batch_mask, neginf)
    else:
      return x

  def batch_heads(self, x, bs, sl, nh, hd):
    return x.view(bs, sl, nh, hd).transpose(1, 2).contiguous().view(bs * nh, sl, hd)

  def unbatch_heads(self, x, bs, sl, od):
    return x.transpose(0, 1).contiguous().view(sl, bs, od).transpose(0, 1)

  def forward(self, input_sequence):
    bs, sl, od = input_sequence.size()
    nh = self.num_heads
    hd = od // nh
    q, k, v = self.projection(input_sequence).chunk(3, dim=-1)
    q *= self.scale
    q, k, v = [self.batch_heads(x, bs, sl, nh, hd) for x in [q, k, v]]
    attention_weights = self.softmax(self.apply_masks(torch.bmm(q, k.transpose(1, 2)), bs, nh, sl, hd))
    return self.out(self.unbatch_heads(torch.bmm(attention_weights, v), bs, sl, od))


class Residual(torch.nn.Module):

  def __init__(self, function, dimension, dropout=0.2):
    super().__init__()
    self.operations = torch.nn.Sequential(
      torch.nn.LayerNorm(dimension),
      function,
      torch.nn.Dropout(dropout)
    )

  def forward(self, input_sequence):
    return self.operations(input_sequence) + input_sequence


class Layer(torch.nn.Module):

  def __init__(self, outer_dimension, inner_dimension, 
      num_heads=8, dropout=0.2):
    super().__init__()
    self.operations = torch.nn.Sequential(
      Residual(Attention(outer_dimension, num_heads), 
        outer_dimension, dropout=dropout),
      Residual(Feedforward(outer_dimension, inner_dimension), 
        outer_dimension, dropout=dropout)
    )

  def forward(self, input_sequence):
    return self.operations(input_sequence)
    

class Transformer(torch.nn.Module):

  def __init__(self, max_sequence_length, num_layers, outer_dimension, inner_dimension, num_heads, dropout):
    super().__init__()
    self.embed = torch.nn.Embedding(256, outer_dimension)
    self.position = torch.nn.Embedding(max_sequence_length, outer_dimension)
    layers = [Layer(outer_dimension, inner_dimension, 
      num_heads=num_heads, dropout=dropout) 
      for i in range(num_layers)]
    self.layers = torch.nn.Sequential(*layers)
    self.norm = torch.nn.LayerNorm(outer_dimension)
    self.out = torch.nn.Linear(outer_dimension, 256)
    self.xent = torch.nn.CrossEntropyLoss()

  def set_batch_mask(self, mask):
    for layer in self.layers:
      layer.operations[0].operations[1].batch_mask = mask

  def forward(self, input, positions, target=None, batch_mask=None):
    if batch_mask is not None:
      self.set_batch_mask(batch_mask)
    embedding = self.embed(input) + self.position(positions)
    scores = self.out(self.norm(self.layers(embedding)))
    loss = self.xent(scores.view(-1, 256), target.view(-1))
    return loss
