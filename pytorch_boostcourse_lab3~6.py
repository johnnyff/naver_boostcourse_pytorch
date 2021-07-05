# -*- coding: utf-8 -*-
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

# +
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#Gradient descent

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1)

lr = 0.1

epochs = 10
for epoch in range(1,epochs+1):
    hypothesis = x_train *W
    cost = torch.mean((hypothesis - y_train)**2)
    gradient = torch.sum((W*x_train -y_train) * x_train)
    

    print("Epoch {:4d}/{} W: {: .3f}, Cost : {:.6f}".format(epoch, epochs, W.item(),cost.item()) )
    W -= lr*gradient

# +
#Using torch.optim

x_train = torch.FloatTensor([[1],[2],[3]])
y_train = torch.FloatTensor([[1],[2],[3]])

W = torch.zeros(1, requires_grad =True)
optimizer = torch.optim.SGD([W],lr =0.15)


epochs = 10
for epoch in range(1,epochs+1):
    hypothesis = x_train *W
    cost = torch.mean((hypothesis - y_train)**2)
    

    print("Epoch {:4d}/{} W: {: .3f}, Cost : {:.6f}".format(epoch, epochs, W.item(),cost.item()) )
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

# +
# 데이터



x_train = torch.FloatTensor([[73, 80, 75],
 [93, 88, 93],
 [89, 91, 90],
 [96, 98, 100],
 [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
# optimizer 설정
optimizer = torch.optim.SGD([W, b], lr=1e-5)


# +
## using nn.module

class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)
    def forward(self, x):
        return self.linear(x)


# 데이터
x_train = torch.FloatTensor([[73, 80, 75],
 [93, 88, 93],
 [89, 91, 90],
 [96, 98, 100],
 [73, 66, 70]])
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])
# 모델 초기화
W = torch.zeros((3, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
model = MultivariateLinearRegressionModel()
# optimizer 설정
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)
nb_epochs = 20
for epoch in range(nb_epochs + 1):

 # H(x) 계산
    Hypothesis = model(x_train)
     # cost 계산
    cost = F.mse_loss(Hypothesis, y_train)
     # cost로 H(x) 개선
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    print('Epoch {:4d}/{} hypothesis: {} Cost: {:.6f}'.format(
    epoch, nb_epochs, Hypothesis.squeeze().detach(),
    cost.item()
     ))


# +
#Logistic regression
# -

torch.manual_seed(1)

# +
x_data = [[1,2],[2,3],[3,1],[4,3],[5,3],[6,2]]
y_data = [[0],[0],[0],[1],[1],[1]]

x_train = torch.FloatTensor(x_data)
y_train = torch.FloatTensor(y_data)

print(x_train.shape)
print(y_train.shape)

# +
W = torch.zeros((2,1),requires_grad =True)
b = torch.zeros(1, requires_grad = True)

optimizer = optim.SGD([W,b], lr =1)

epochs = 1000

for epoch in range(epochs+1):
    hypothesis = 1/ (1+torch.exp(-(x_train.matmul(W)+b)))
    cost = F.binary_cross_entropy(hypothesis,y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    if epoch %100 ==0:
        print("Epoch {:4d}/{} Cost : {:.6f}".format(epoch, epochs, cost.item()))


# +
#higher implementation with class

class BinaryClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)
        self.sigmoid = nn.Sigmoid()
    def forward(self,x):
        return self.sigmoid(self.linear(x))
    
model = BinaryClassifier()

optimizer = optim.SGD(model.parameters(),lr =1)
epochs = 100
for epoch in range(epochs+1):
    #Calculate H(X)
    hypothesis = model(x_train)
    
    #calculate cost
    cost = F.binary_cross_entropy(hypothesis, y_train)
    
    # update H(x) by cost
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch %10 ==0:
        prediction = hypothesis >= torch.FloatTensor([0.5])
        correct_prediction = prediction.float() == y_train
        accuracy = correct_prediction.sum().item()/len(correct_prediction)
        print("Epoch {:4d}/{} Cost : {:.6f} Accurach {:2.2f}%".format(epoch, epochs, cost.item(),accuracy*100))        


# +
#Softmax Classification

#Training with low-level corss Entropy Loss
x_train = [[1,2,1,1],
          [2,1,3,2],
          [3,1,3,4],
          [4,1,5,5],
          [1,7,5,5],
          [1,2,5,6],
          [1,6,6,6],
          [1,7,7,7]]
y_train = [2,2,2,1,1,1,0,0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

W= torch.zeros((4,3), requires_grad =True)
b = torch.zeros(1, requires_grad = True)

optimizer= optim.SGD([W,b], lr = 0.1)

epochs = 1000
for epoch in range(epochs+1):
    z = x_train.matmul(W)+b
    cost = F.cross_entropy(z, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch %100 ==0:
        print("Epoch {:4d}/{} Cost : {:.6f}".format(epoch, epochs, cost.item()))  

        
#high-level implementation

class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4,3) 
    def forward(self,x):
        return self.linear(x)
    
model = SoftmaxClassifierModel()

optimizer= optim.SGD(model.parameters(), lr = 0.1)

epochs = 1000
for epoch in range(epochs+1):
    prediction = model(x_train)
    cost = F.cross_entropy(prediction, y_train)
    
    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
    
    if epoch %100 ==0:
        print("Epoch {:4d}/{} Cost : {:.6f}".format(epoch, epochs, cost.item()))  

# -

prediction




