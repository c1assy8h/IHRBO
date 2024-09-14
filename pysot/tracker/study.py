import torch
a=torch.rand((1,10,25,25))
b=a.permute(1, 2, 3, 0).contiguous().view(2, -1) #[2,3125]
c=b.permute(1, 0)
print(c.shape)