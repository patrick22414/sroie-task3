import torch
from torch import nn

class Model0(nn.Module):
    def __init__(self, vocab_size, embed_size, hidd_size):
        self.embed = nn.Embedding(vocab_size, embed_size)
