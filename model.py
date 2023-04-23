import torch
from torch import nn

from collections import OrderedDict

class Block(nn.Module):
    def __init__(self, in_size, out_size, drop_p):
        super(Block, self).__init__()
        self.block = nn.Sequential(OrderedDict([
            ('lin', nn.Linear(in_size, out_size)),
            ('act', nn.ReLU()),
            ('drop', nn.Dropout(drop_p))
        ]))
    def forward(self, x):
        return self.block(x)
    
class BlockList(nn.Module):
    def __init__(self, count, in_size, hid_size1=312, out_size=312, drop_p=0.25):
        super(BlockList, self).__init__()
        self.list = nn.ModuleList([])
        if count == 0:
            return 
        
        if count == 1:
            self.list.append(Block(in_size, out_size, drop_p))
            return 
        
        self.list.append(Block(in_size, hid_size1, drop_p))
        for _ in range(count - 2):
            self.list.append(Block(hid_size1, hid_size1, drop_p))
        self.list.append(Block(hid_size1, out_size, drop_p)) 
        
    def forward(self, x):
        for layer in self.list:
            x = layer(x)
        return x

class gptKILLER(nn.Module):
    def __init__(self, pipeline, frozen_bert, hid_size, count_of_blocks):
        super(gptKILLER, self).__init__()
        self.model = pipeline.model
        
        if frozen_bert:
            for param in self.model.bert.parameters():
                param.requires_grad = False
        else:
            for param in self.model.bert.embeddings.parameters():
                param.requires_grad = False
            
        hid_size1 = hid_size
        hid_size2 = 312
        drop_p = 0.25
        self.model.qa_outputs = nn.Sequential(OrderedDict([
            ('blockList', BlockList(count_of_blocks, 312, hid_size1, hid_size2, drop_p)),
            ('fc', nn.Linear(hid_size2, 2))
        ]))
        
    def forward(self, question, types, attention_mask, start, end):
        return self.model(
                    question, 
                    token_type_ids=types,
                    attention_mask=attention_mask,
                    start_positions=start,
                    end_positions=end
        )

def create_model_and_optimizer(model_class, model_params, device, lr=1e-3, beta1=0.9, beta2=0.999):
    model = model_class(**model_params)
    model = model.to(device)
    
    optimized_params = []
    for param in model.parameters():
        if param.requires_grad:
            optimized_params.append(param)
    optimizer = torch.optim.Adam(optimized_params, lr, [beta1, beta2])
    return model, optimizer