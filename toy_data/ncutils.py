import torch
from torch.utils.data import Dataset
import numpy as np


class NumCharDataset(Dataset):

  def __init__(self, x, t):
    super().__init__()
    self.x = x
    self.t = t


  def __getitem__(self, idx):
    return self.x[idx], self.t[idx]


  def __len__(self):
    return self.x.shape[0]



class NumCharCorpus:

  def __init__(self, file_name='addition.txt'):
    self.corpus = None
    self.vocab = None
    self.char_to_id = {}
    self.id_to_char = {}
    self.x = None
    self.t = None
    self._load(file_name)
    self._update_map()
    self._build_dataset()


  def _load(self, file_name):
    with open(file_name, 'r') as f:
      txt = f.read()
    self.corpus = txt.split('\n')
    self.vocab = set(txt.replace('\n', ''))


  def _update_map(self):
    for i, c in enumerate(self.vocab):
      self.char_to_id[c] = i
      self.id_to_char[i] = c


  def _build_dataset(self):
    ids = []
    for s in self.corpus:
      ids.append(self._build_seq(s))
    ids = torch.tensor(ids, dtype=torch.int)
    self.x = ids[:, :7].clone()
    self.t = ids[:, 7:].clone().long()


  def _build_seq(self, seq):
    ids = []
    for c in seq:
      ids.append(self.char_to_id[c])
    return ids


  def get_dict(self):
      return self.char_to_id, self.id_to_char


  def fliplr_x(self):
    self.x = torch.fliplr(self.x)


  def shuffle(self, seed=1984):
    indices = np.arange(len(self.x))
    if seed is not None:
      np.random.seed(seed)
    np.random.shuffle(indices)
    self.x = self.x[indices]
    self.t = self.t[indices]


  def get_dataset(self, train=True):
    split_at = len(self.x) - len(self.x) // 10
    if train:
      return NumCharDataset(self.x[:split_at], self.t[:split_at])
    else:
      return NumCharDataset(self.x[split_at:], self.t[split_at:])


  @property
  def vocab_size(self):
    return len(self.vocab)


  def to_str(self, ids):
    s = []
    for id in ids:
      tmp = self.id_to_char[id]
      s.append(tmp)
    return "".join(s)

