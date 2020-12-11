import torch.nn.functional as F 
import torch
import torch.nn as nn
import pdb


class CompactLoss(nn.Module):

    def __init__(self, loss=(lambda x: torch.sigmoid(-x))):
        super(CompactLoss,self).__init__()
        self.positive = 0
        self.negative = 1
        self.min_count = torch.tensor(1.)
        self.positive_center = torch.tensor(0.)
        self.positive_num = 0

    
    def forward(self, inp, target, test=False):
        #assert(inp.shape == target.shape)     
        positive, negative = target == self.positive, target == self.negative
        positive, negative = positive.type(torch.float), negative.type(torch.float)
        
        if inp.shape != target.shape:
            inp = self._avg_pooling(inp).view(inp.size(0),inp.size(1))
            inp = torch.mean(inp,dim=1)

        if inp.is_cuda:
            self.min_count = self.min_count.cuda()
        n_positive, n_negative = torch.max(self.min_count, torch.sum(positive)), torch.max(self.min_count, torch.sum(negative))
        
        positive_feat = positive*inp
        negtive_feat = negative*inp
        positive_feat = positive_feat[positive_feat!=0.0]
        negtive_feat = negtive_feat[negtive_feat!=0.0]
        self.positive_center.data = (torch.sum(positive_feat)+self.positive_center*self.positive_num) / (self.positive_num+n_positive)
        self.positive_num+=n_positive
        positive_loss = torch.sum((positive_feat - self.positive_center)**2)
        negtive_loss = -torch.sum((negtive_feat - self.positive_center)**2)
        return positive_loss+negtive_loss
       