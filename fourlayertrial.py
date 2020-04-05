import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import random

# XOR 문제를 해결하기 위해 dataset 만들기.
X_data = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]]).float()
y_data = torch.tensor([0, 1, 1, 0]).float()

"""
NPUT	OUTPUT
A	B	A XOR B
0	0	0
0	1	1
1	0	1
1	1	0
"""

class Model(nn.Module):

    def __init__(self, input_size, H1, output_size):
        super().__init__()
        ###############################################################################################################
        #                  TODO : Forward path를 위한 Linear함수 또는 Weight 와 bias를 정의                           #
        ###############################################################################################################

        self.seed = torch.manual_seed(1393204) # make random num

        # self.lin1 = nn.Linear(input_size, output_size) # for layer 1

        self.lin1 = nn.Linear(input_size, H1) # Affine layer(input size, output_size), for layer 2 and 4
        # self.lin2 = nn.Linear(input_size, output_size)"""

        self.lin2 = torch.nn.Linear(H1, H1) # for layer 4
        self.lin3 = torch.nn.Linear(H1, H1)
        self.lin4 = torch.nn.Linear(H1, output_size)
        #torch.nn.init.xavier_uniform(self.lin1.weight) # weight initialzation by Xavier
        #torch.nn.init.xavier_uniform(self.lin2.weight)
        #torch.nn.init.xavier_uniform(self.lin3.weight)
        #torch.nn.init.xavier_uniform(self.lin4.weight) 

        #torch.nn.init.xavier_normal(self.lin1.weight) # weight initialzation by Xavier
        #torch.nn.init.xavier_normal(self.lin2.weight) 
        #torch.nn.init.xavier_normal(self.lin3.weight) 
        #torch.nn.init.xavier_normal(self.lin4.weight) 

        torch.nn.init.uniform_(self.lin1.weight, 0, 0.01)
        torch.nn.init.uniform_(self.lin2.weight, 0, 0.01)
        torch.nn.init.uniform_(self.lin3.weight, 0, 0.01)
        torch.nn.init.uniform_(self.lin4.weight, 0, 0.01)

        #torch.nn.init.constant_(self.lin1.weight, 0.001)
        #torch.nn.init.constant_(self.lin2.weight, 0.001)
        #torch.nn.init.constant_(self.lin3.weight, -0.001)
        #torch.nn.init.constant_(self.lin4.weight, -0.001)

        self.bn1 = nn.BatchNorm1d(H1) # (batch) normalization after affine
        self.bn2 = nn.BatchNorm1d(output_size)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        # TODO 1. Linear - Sigmoid 의 구조를 가지는 Forward path를 수행하고 결과를 x에 저장                          #

        # TODO 2. Linear - Sigmoid - Linear - Sigmoid 의 구조를 가지는 Forward path를 수행하고 결과를 x에 저장       #

        # TODO 3. Linear - Linear - Linear - Linear - Sigmoid 의 구조를 가지는 Forward path를 수행하고 결과를 x에 저장#
        ###############################################################################################################
        # x = torch.sigmoid(self.lin1(x)) # layer 1

        # x = torch.sigmoid(self.bn1(self.lin1(x))) # layer 2
        # x = torch.sigmoid(self.lin2(x))

        x = self.bn1(self.lin1(x)) # layer 4
        x = self.bn1(self.lin2(x))
        x = self.bn1(self.lin3(x))
        x = self.bn2(self.lin4(x))
        x = torch.sigmoid(x)
        # x = torch.nn.functional.softmax(x) # none
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

        return x

    def predict(self, x):
        return self.forward(x) >= 0.5

    def loss(self, x, t):
        y = self.predict(x)
        return self

model = Model(2, 2, 1)

##############################################################################################################
#                  TODO : 손실함수(BCELoss)와 optimizer(Adam)를 정의(learning rate=0.01)                     #
##############################################################################################################
BCELoss = nn.BCELoss() # cross entropy for calculating loss func
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01) # for learning
###############################################################################################################
#                                              END OF YOUR CODE                                               #
###############################################################################################################

epochs = 200000 # how many
losses = [] # store record of loss

# batch train step
for i in range(epochs):
    loss=None
    ##############################################################################################################
    #                         TODO : foward path를 진행하고 손실을 loss에 저장                                   #
    #                               전체 data를 모두 보고 updata를 진행하는 Batch gradient descent(BGD)을 진행   #
    ##############################################################################################################
    y_pred = model.forward(X_data) # forward (Affine)
    loss = BCELoss(y_pred, y_data) # calculate loss
    ###############################################################################################################
    #                                              END OF YOUR CODE                                               #
    ###############################################################################################################

    print("epochs:", i, "loss:", loss.item())
    losses.append(loss.item()) # store loss in losses list
    ##############################################################################################################
    #                     TODO : optimizer를 초기화하고 gradient를 계산 후 model을 optimizing                    #
    ##############################################################################################################
    optimizer.zero_grad() # make variation 0
    loss.backward() # do the backpropagation
    optimizer.step() # learning or update weight
    ###############################################################################################################
    #                                              END OF YOUR CODE                                               #
    ###############################################################################################################


def cal_score(X, y):
    y_pred = model.predict(X)
    score = float(torch.sum(y_pred.squeeze(-1) == y.byte())) / y.shape[0]

    return score


print('test score :', cal_score(X_data, y_data))
plt.plot(range(epochs), losses)
plt.show()


def plot_decision_boundray(X):
    x_span = np.linspace(min(X[:, 0]), max(X[:, 0]))
    y_span = np.linspace(min(X[:, 1]), max(X[:, 1]))

    xx, yy = np.meshgrid(x_span, y_span)

    grid = torch.tensor(np.c_[xx.ravel(), yy.ravel()]).float()

    pred_func = model.forward(grid)

    z = pred_func.view(xx.shape).detach().numpy()

    plt.contourf(xx, yy, z)
    plt.show()


plot_decision_boundray(X_data)
