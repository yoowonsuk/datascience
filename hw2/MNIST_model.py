import torch
import torch.nn as nn
import torch.nn.functional as F


class MNIST_model(nn.Module):
    def __init__(self):
        super().__init__()
        ##############################################################################################################
        #                         TODO : 4-layer feedforward 모델 생성 (evaluation report의 세팅을 사용할 것)           #
        ##############################################################################################################
        self.lin1 = nn.Linear(784, 200) # 28 * 28 = 784
        self.lin2 = nn.Linear(200, 200)
        self.lin3 = nn.Linear(200, 200)
        self.lin4 = nn.Linear(150, 10) # the number of 0~9 equals 10

        torch.nn.init.kaiming_normal_(self.lin1.weight)
        torch.nn.init.kaiming_normal_(self.lin2.weight)
        torch.nn.init.kaiming_normal_(self.lin3.weight)
        torch.nn.init.kaiming_normal_(self.lin4.weight)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################

    def forward(self, x):
        ##############################################################################################################
        #                         TODO : forward path 수행, 결과를 x에 저장                                            #
        ##############################################################################################################
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        log_softmax = torch.nn.LogSoftmax(dim=1)
        x = log_softmax(x)
        ###############################################################################################################
        #                                              END OF YOUR CODE                                               #
        ###############################################################################################################
        return x


class Config():
    def __init__(self):
        self.batch_size = 200
        self.lr_adam = 0.0001
        self.lr_adadelta = 0.1
        self.epoch = 100
        self.weight_decay = 1e-03
