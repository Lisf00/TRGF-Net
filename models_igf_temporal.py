import torch
from torch import nn
from models_simple_hgt import SimpleHGT

class TemporalIGF(SimpleHGT):
    def __init__(self,kind):
        super().__init__('simple')
        self.temporal=(nn.GRU if kind=='gru' else nn.LSTM)(64,32,batch_first=True,bidirectional=True)
    def forward(self,x):
        b=x.shape[0];h=self.encoder(x.reshape(b*6,1,-1));h=h.reshape(b,6,16,-1).permute(0,3,1,2)+self.identity
        h=h[:,:,[0,1,4,2,3,5]].reshape(b,h.shape[1],2,3,16);z=self.local(h)
        context=self.cross(z.mean(-2)).unsqueeze(-2).expand_as(z)
        z=z+torch.sigmoid(self.gate(torch.cat([z,context],-1)))*context
        z=self.fuse(z.flatten(2).transpose(1,2))
        z,_=self.temporal(z.transpose(1,2));z=z.transpose(1,2);n=z.shape[-1]
        pooled=torch.stack([z[:,:,i*n//8:((i+1)*n+7)//8].mean(-1) for i in range(8)],-1).flatten(1)
        return self.classifier(pooled)

def make_model(v):
    if v=='igf':return SimpleHGT('simple')
    if v in ('gru','lstm'):return TemporalIGF(v)
    raise ValueError(v)
