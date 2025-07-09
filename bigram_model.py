import torch
import torch.nn as nn
from torch.nn import functional as F


# HYPER PARAMETERS

batch_size = 32
block_size = 8
max_iters = 3000
eval_interval = 300
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
#----------------------------------------------------
torch.manual_seed(1337)


# wget.download('https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt')
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# GETTING ALL THE UNQUE CHARACTERS THAT OCCUR IN THE TEXT

chars = sorted(list(set(text)))
vocab_size = len(chars)
print(''.join(chars))
print(vocab_size)


