import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5_model(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################################################
        #                         TODO : LeNet5 모델 생성                                                             #
        ##############################################################################################################
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.lin1 = nn.Linear(16*5*5, 120)
        self.lin2 = nn.Linear(120, 84)
        self.lin3 = nn.Linear(84, 10)
        self.pooling = nn.MaxPool2d(kernel_size = 2, stride = 2)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        x = torch.relu(self.conv1(x))
        x = self.pooling(x)
        x = torch.relu(self.conv2(x))
        x = self.pooling(x)
        x = x.view(-1, 16*5*5)
        x = torch.relu(self.lin1(x))
        x = torch.relu(self.lin2(x))
        x = self.lin3(x)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
        return x


