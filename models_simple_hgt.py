import torch
from torch import nn

class Graph(nn.Module):
    def __init__(self,d=16):
        super().__init__();self.q=nn.Linear(d,d);self.k=nn.Linear(d,d);self.v=nn.Linear(d,d);self.norm=nn.LayerNorm(d)
    def forward(self,x):
        a=(self.q(x)@self.k(x).transpose(-1,-2)/x.shape[-1]**.5).softmax(-1)
        return self.norm(x+a@self.v(x))

class SimpleHGT(nn.Module):
    def __init__(self,variant='simple'):
        super().__init__();self.variant=variant
        self.encoder=nn.Sequential(nn.Conv1d(1,8,9,4,4),nn.GroupNorm(2,8),nn.GELU(),nn.Conv1d(8,16,7,4,3),nn.GroupNorm(4,16),nn.GELU(),nn.Conv1d(16,16,5,2,2),nn.GELU())
        self.identity=nn.Parameter(torch.zeros(1,1,6,16))
        self.local=nn.Identity() if variant=='no_local' else Graph()
        if variant!='no_cross':self.cross=Graph();self.gate=nn.Linear(32,16)
        self.fuse=nn.Conv1d(96,64,1)
        self.classifier=nn.Sequential(nn.Dropout(.15),nn.Linear(512,23))
    def forward(self,x):
        b=x.shape[0];h=self.encoder(x.reshape(b*6,1,-1));h=h.reshape(b,6,16,-1).permute(0,3,1,2)+self.identity
        h=h[:,:,[0,1,4,2,3,5]].reshape(b,h.shape[1],2,3,16);z=self.local(h)
        if self.variant!='no_cross':
            context=self.cross(z.mean(-2)).unsqueeze(-2).expand_as(z)
            z=z+torch.sigmoid(self.gate(torch.cat([z,context],-1)))*context
        z=self.fuse(z.flatten(2).transpose(1,2));n=z.shape[-1]
        pooled=torch.stack([z[:,:,i*n//8:((i+1)*n+7)//8].mean(-1) for i in range(8)],-1).flatten(1)
        return self.classifier(pooled)
