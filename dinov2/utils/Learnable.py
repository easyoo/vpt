import torch
import torch.nn as nn

class Learnable(nn.Module):
    def __init__(self, 
                 layers = 12,
                 dim = 768,
                 learnable_token_per_layer = 4):
        super().__init__()
        self.token_dict = nn.ParameterDict({
            str(i): nn.Parameter(torch.randn(1,learnable_token_per_layer, dim))
            for i in range(layers)
        })
