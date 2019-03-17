import torch
import torchfile

Readed_t7 = torchfile.load('Black_Footed_Albatross_0003_796136.t7')
print(Readed_t7[b'word'].shape)
