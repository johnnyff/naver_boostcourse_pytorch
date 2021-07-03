# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: pytorch
#     language: python
#     name: pytorch
# ---

import numpy as np
import torch

# ### Numpy

t = np.array([0.,1.,2.,3.,4.,5.,6.])
print(t)

print("Rank of t : ", t.ndim)
print("shape of t : ", t.shape)

t = np.array([[1., 2.,3.],[4.,5.,6.,],[7.,8.,9]])
print(t)

# ### Pytorch tensor

t = torch.FloatTensor([0.,1.,2.,3.,4.,5.,6.])
print(t)

print(t.dim())  #rank
print(t.shape)  #shape
print(t.size())  #shape

# +
# Vector + scalar
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([3]) # 3 -> [[3, 3]]
print(m1 + m2)

# 2 x 1 Vector + 1 x 2 Vector
m1 = torch.FloatTensor([[1, 2]])
m2 = torch.FloatTensor([[3], [4]])
print(m1 + m2)

# +
# Mul vs Matmul

m1 = torch.FloatTensor([[1,2],[3,4]])
m2 = torch.FloatTensor([[1],[2]])
print(m1.matmul(m2))
print("---------------------")
print(m1*m2)
# -

t= torch.FloatTensor([[1,2],[3,4]])
print(t)

print(t.mean())
print(t.mean(dim=0))
print(t.mean(dim=1))
print(t.mean(dim=-1))

print(t.sum())
print(t.sum(dim=0))
print(t.sum(dim=1))
print(t.sum(dim=-1))

print(t.max())

#Max and Argmax
print(t.max(dim=0))
print("Max : ", t.max(dim=0)[0])
print("ArgMax : ", t.max(dim=0)[1])

# +
#view

t = np.array([[[0,1,2],
              [3,4,5]],
             [[6,7,8,],
             [9,10,11]]])
ft = torch.FloatTensor(t)
print(t.shape)
# -

print(ft.view([-1,3]))
print(ft.view([-1,3]).shape)

# +
#Squeeze

ft = torch.FloatTensor([[0],[1],[2]])
print(ft)
print(ft.shape)
# -

print(ft.squeeze())
print(ft.squeeze().shape)

# +
#Unsqueeze
ft = torch.Tensor([0,1,2])
print(ft.shape)

print(ft.unsqueeze(0))
print(ft.unsqueeze(0).shape)

print(ft.unsqueeze(1))
print(ft.unsqueeze(1).shape)

# +
#Casting
lt = torch.LongTensor([1, 2, 3, 4])
print(lt)
print(lt.float())

bt = torch.ByteTensor([True, False, False, True])
print(bt)
print(bt.long())
print(bt.float())


# +
#concatenate

x= torch.FloatTensor([[1,2],[3,4]])
y= torch.FloatTensor([[5,6],[7,8]])

print(torch.cat([x,y], dim=0))
print(torch.cat([x,y], dim=1))

# +
#stacking
x = torch.FloatTensor([1, 4])
y = torch.FloatTensor([2, 5])
z = torch.FloatTensor([3, 6])

print(torch.stack([x, y, z]))
print(torch.stack([x, y, z], dim=1))

# -

print(torch.cat([x.unsqueeze(0),y.unsqueeze(0),z.unsqueeze(0)], dim=0))

# +
#in-place operation

x= torch.FloatTensor([[1,2],[3,4]])

print(x.mul(2.))
print(x)
print(x.mul_(2.))
print(x)
# -


