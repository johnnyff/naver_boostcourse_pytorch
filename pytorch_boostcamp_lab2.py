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

# +
# linear regression
x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[4],[5],[6]])

w = torch.zeros(1, requires_grad = True)
b = torch.zeros(1, requires_grad = True)


# +
optimizer =  torch.optim.SGD([w,b], lr =0.01)

epochs = 1000

for epoch in range(1,epochs+1):
    hypothesis = x_train *w +b
    cost = torch.mean((hypothesis - y_train)**2)
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
# -

print("w : ",w ,' b : ',b)




