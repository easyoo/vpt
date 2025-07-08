from util.agent_pvt import Block
import torch
block = Block(768,784+4,2)
x = torch.randn(1, 784+4, 768)
y= block(x,28,28)
print(y.shape)